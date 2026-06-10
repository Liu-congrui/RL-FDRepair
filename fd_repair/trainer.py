"""
训练模块
========
两阶段训练：
  1. 监督预训练（Supervised Pretraining）
  2. PPO强化学习微调
"""

from __future__ import annotations

import os
import time
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .fd_utils import (
    FD, ConflictGroup,
    build_all_conflict_groups, count_fd_violations,
    rank_lhs_alts_by_evidence,
    build_fd_graph, get_neighborhood, get_all_cg_rows,
    compute_row_conflict_counts,
)
from .features import FeatureExtractor, FEATURE_DIM
from .environment import FDRepairEnv, RowLockRepairEnv, TOTAL_ACTIONS, N_RHS_ACTIONS, N_LHS_ACTIONS
from .policy import ActorCritic, RolloutBuffer
from .evaluator import compute_metrics, compute_episode_metrics
from .error_injection import InjectionResult, inject_errors
from config import TRAIN_CONFIG, REWARD_CONFIG, ENV_CONFIG, ROW_LOCK_CONFIG, NETWORK_CONFIG


# ============================================================
# 辅助：为单个冲突组生成最优动作标签（监督预训练用）
# ============================================================

def generate_pretrain_label(
    cg: ConflictGroup,
    row_idx: int,
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    all_fds: Optional[List[FD]] = None,
) -> Optional[int]:
    """
    对 CG 中指定行生成监督标签（最优动作 index）。

    动作编号：0/1/2=RHS_TOP3, 3/4/5=LHS_TOP3, 6=NO_OP

    逐行判断（基于 GT）：
      1. 该行 LHS 值错误 → LHS_EJECT（在 top3 中找到正确 LHS）
      2. 该行 RHS 值错误（≠ majority）且 RHS 候选可用 → RHS_UNIFY(action=0)
      3. 该行完全正确 → NO_OP
    """
    from collections import Counter

    if not cg.has_conflict:
        return TOTAL_ACTIONS - 1  # NO_OP

    fd = cg.fd
    col_map_clean = {c.lower(): c for c in clean_df.columns}
    col_map_dirty = {c.lower(): c for c in dirty_df.columns}

    lhs_cols_clean = [col if col in clean_df.columns else col_map_clean.get(col.lower(), col) for col in fd.lhs_cols]
    lhs_cols_dirty = [col if col in dirty_df.columns else col_map_dirty.get(col.lower(), col) for col in fd.lhs_cols]

    # 检查该行的 LHS 是否错误
    dirty_lhs = tuple(dirty_df.at[row_idx, col] for col in lhs_cols_dirty if col in dirty_df.columns)
    clean_lhs = tuple(clean_df.at[row_idx, col] for col in lhs_cols_clean if col in clean_df.columns)
    lhs_is_wrong = (dirty_lhs != clean_lhs)

    # 检查该行的 RHS 是否错误
    col_map_d = {c.lower(): c for c in dirty_df.columns}
    rhs_col_d = cg.fd.rhs_col if cg.fd.rhs_col in dirty_df.columns else col_map_d.get(cg.fd.rhs_col.lower())
    rhs_is_wrong = False
    if rhs_col_d and rhs_col_d in dirty_df.columns:
        dirty_rhs = dirty_df.at[row_idx, rhs_col_d]
        clean_rhs = clean_df.at[row_idx, rhs_col_d]
        rhs_is_wrong = (str(dirty_rhs) != str(clean_rhs))

    if lhs_is_wrong and cg.lhs_alternatives:
        # LHS 错误 → 找正确 LHS 在 top3 中的位置
        if rhs_col_d:
            majority_rhs = Counter(dirty_df.loc[cg.row_indices, rhs_col_d].tolist()).most_common(1)[0][0]
            eject_candidates = [i for i in cg.row_indices if dirty_df.at[i, rhs_col_d] != majority_rhs]
        else:
            eject_candidates = []

        if not eject_candidates:
            eject_candidates = cg.row_indices[:1]
        eject_row = eject_candidates[0]

        fds_for_rank = all_fds if all_fds else [cg.fd]
        lhs_alts_list = rank_lhs_alts_by_evidence(
            dirty_df, cg.fd, fds_for_rank, cg.lhs_alternatives, eject_row
        )

        for i, (alt_lhs, _, _) in enumerate(lhs_alts_list):
            if alt_lhs == clean_lhs and i < N_LHS_ACTIONS:
                return N_RHS_ACTIONS + i
        # 正确 LHS 不在 top3 → 退回到 RHS_UNIFY
        return 0

    if rhs_is_wrong and len(cg.rhs_candidates) > 0:
        # RHS 错误 → 在 top3 中找到 GT 正确的 RHS 候选，返回其索引
        rhs_list = cg.rhs_candidate_list()
        for i, rhs_val in enumerate(rhs_list):
            if i >= N_RHS_ACTIONS:
                break
            if str(rhs_val) == str(clean_rhs):
                return i
        # 正确值不在 top3 → 退回到 action 0（多数值）
        return 0

    if not lhs_is_wrong and not rhs_is_wrong:
        # 该行完全正确 → NO_OP
        return TOTAL_ACTIONS - 1

    # 兜底
    return 0


# ============================================================
# 监督预训练
# ============================================================

def pretrain(
    model: ActorCritic,
    injection_result: InjectionResult,
    fds: List[FD],
    device: str = "cuda",
    config: Dict = None,
) -> List[float]:
    """监督预训练阶段：逐行生成标签，与 PPO 阶段特征分布一致。"""
    cfg = config or TRAIN_CONFIG
    model.to(device)

    dirty_df = injection_result.dirty_df
    clean_df = injection_result.clean_df

    # FeatureExtractor 用脏数据做全局统计（保留特征区分力）
    fe = FeatureExtractor(global_df=dirty_df, fds=fds, raw_df=dirty_df)

    obs_list, label_list = [], []

    col_map = {c.lower(): c for c in dirty_df.columns}
    max_steps = ENV_CONFIG["max_steps_per_episode"]
    conflict_counts = compute_row_conflict_counts(dirty_df, fds)

    all_cgs_flat = []
    for fd, cgs in build_all_conflict_groups(dirty_df, fds):
        for cg in cgs:
            all_cgs_flat.append((fd, cg))

    total_cgs = len(all_cgs_flat)
    print(f"[Pretrain] Generating samples from {total_cgs} CGs...")

    for cg_idx, (fd, cg) in enumerate(all_cgs_flat):
        if (cg_idx + 1) % 50 == 0 or cg_idx == 0:
            print(f"  Pretrain sample gen: {cg_idx+1}/{total_cgs} CGs, {len(obs_list)} samples so far")

        all_rows = get_all_cg_rows(cg, dirty_df)
        if not all_rows:
            continue

        # 计算 CG 级 base obs（行无关特征相同）
        base_obs = fe.extract(cg, dirty_df, step=0, max_steps=max_steps)

        for row_idx in all_rows:
            label = generate_pretrain_label(cg, row_idx, dirty_df, clean_df, all_fds=fds)
            if label is None:
                continue

            # 替换最后2维：冲突总数 + CEC（pretrain 阶段 CEC 取中立值）
            obs = base_obs.copy()
            obs[-2] = float(conflict_counts.get(row_idx, 0)) / 50.0
            obs[-1] = 0.5  # CEC

            obs_list.append(obs)
            label_list.append(label)

    if not obs_list:
        print("[Pretrain] No training samples found. Skipping pretrain.")
        return []

    obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
    label_tensor = torch.LongTensor(label_list).to(device)

    loader = DataLoader(TensorDataset(obs_tensor, label_tensor),
                        batch_size=cfg["pretrain_batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg["pretrain_lr"])
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    print(f"[Pretrain] Training on {len(obs_list)} samples for {cfg['pretrain_epochs']} epochs...")

    for epoch in range(cfg["pretrain_epochs"]):
        epoch_loss = 0.0
        for batch_obs, batch_labels in loader:
            logits = model._compute_logits(batch_obs)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  [Pretrain] Epoch {epoch+1}/{cfg['pretrain_epochs']}, Loss: {avg_loss:.4f}")

    print("[Pretrain] Done.")
    return loss_history


# ============================================================
# PPO训练
# ============================================================

class PPOTrainer:
    """PPO训练器（单数据集）。"""

    def __init__(
        self,
        model: ActorCritic,
        injection_result: InjectionResult,
        fds: List[FD],
        device: str = "cuda",
        config: Dict = None,
        reward_config: Dict = None,
        lhs_ratio: float = None,
    ):
        self.model = model.to(device)
        self.injection_result = injection_result
        self.fds = fds
        self.device = device
        self.cfg = config or TRAIN_CONFIG
        self.reward_config = reward_config
        self.lhs_ratio = lhs_ratio

        self._raw_model = model.module if hasattr(model, "module") else model

        self.dirty_df = injection_result.dirty_df
        self.clean_df = injection_result.clean_df

        # FeatureExtractor 用脏数据做全局统计（特征有区分力）
        # clean_df 用伪数据（仅用于 correctness reward）
        self.fe = FeatureExtractor(global_df=self.dirty_df, fds=fds, raw_df=self.dirty_df)

        self.all_cgs: List[Tuple[FD, ConflictGroup]] = []
        for fd, cgs in build_all_conflict_groups(self.dirty_df, fds):
            for cg in cgs:
                self.all_cgs.append((fd, cg))

        print(f"[PPO] Total conflict groups available: {len(self.all_cgs)}")

        from .fd_utils import count_all_fd_violations as _cav
        self._initial_global_viol_total = sum(_cav(self.dirty_df, fds).values())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg["ppo_lr"])
        self._current_entropy_coef = self.cfg["ppo_entropy_coef"]  # 退火用
        self.buffer = RolloutBuffer(
            buffer_size=self.cfg["ppo_steps_per_epoch"],
            obs_dim=FEATURE_DIM,
            n_actions=TOTAL_ACTIONS,
            device=device,
        )
        self.metrics_history: List[Dict] = []
        os.makedirs(self.cfg["checkpoint_dir"], exist_ok=True)

    def train(self) -> List[Dict]:
        print(f"[PPO] Starting training for {self.cfg['ppo_epochs']} epochs...")

        best_f1 = -1.0
        best_epoch = -1
        best_model_state = None

        # 使用伪干净数据作为注错源（不接触真实GT）
        base_clean_df = self.clean_df
        fd_graph = build_fd_graph(self.fds)
        train_error_rates = self.cfg.get("train_error_rates", [0.2])
        print(f"[PPO] Training with per-epoch error injection: rates={train_error_rates}")

        for epoch in range(self.cfg["ppo_epochs"]):
            t0 = time.time()
            self.buffer.reset()
            ep_rewards, ep_lengths, ep_w2r, ep_r2w = [], [], [], []
            last_obs = None
            last_value = 0.0

            # ── 每 epoch 随机注错（模拟不同错误率，clean_df 对注入错误 100% 准确）──
            import random as _random
            error_rate = _random.choice(train_error_rates)
            epoch_seed = self.cfg.get("seed", 42) + epoch * 1000
            from .error_injection import inject_errors as _inject
            injection = _inject(base_clean_df, self.fds, seed=epoch_seed,
                               error_rate=error_rate)
            current_df = injection.dirty_df
            epoch_dirty_df = current_df.copy()  # 保存原始脏数据，用于 compute_episode_metrics

            # 重建特征提取器和 CG 缓存（基于新的脏数据）
            self.fe = FeatureExtractor(global_df=current_df, fds=self.fds, raw_df=current_df)

            self.model.eval()

            # ── 内部循环：按冲突数排序遍历行，模型通过 NO_OP 决定跳过 ──
            repair_round = 0
            max_repair_rounds = 200
            action_counts = {i: 0 for i in range(TOTAL_ACTIONS)}
            processed_rows: Set[int] = set()

            # 构建 CG 缓存（epoch 开始时计算，仅在数据修改后重建）
            def _build_cg_cache(df):
                cg_list = []
                for fd, cgs in build_all_conflict_groups(df, self.fds):
                    for cg in cgs:
                        if cg.has_conflict:
                            cg_list.append((fd, cg))
                counts = compute_row_conflict_counts(df, self.fds)
                return cg_list, counts

            cg_cache, conflict_counts = _build_cg_cache(current_df)
            cg_dirty = False

            # ── 课程学习：按 epoch 阶段过滤候选行 ──
            # Phase 1: 只用注错行 → 学会修错行
            # Phase 2: 混入部分正确行 → 学会跳过正确行
            # Phase 3: 全部行 → 精炼判别能力
            error_row_set = set()
            if injection.error_records:
                error_row_set = set(r.row_idx for r in injection.error_records)
            error_only_epochs = self.cfg.get("curriculum_error_only_epochs", 0)
            mix_epochs = self.cfg.get("curriculum_mix_epochs", 0)
            mix_ratio = self.cfg.get("curriculum_mix_correct_ratio", 0.3)

            never_all = self.cfg.get("curriculum_never_all", False)
            if epoch < error_only_epochs:
                curriculum_phase = "errors_only"
            elif never_all or epoch < error_only_epochs + mix_epochs:
                curriculum_phase = "mix"
            else:
                curriculum_phase = "all"

            # 收集所有需要处理的行，按冲突数降序排列
            candidate_rows = []
            seen = set()
            for fd, cg in cg_cache:
                for r in cg.row_indices:
                    if r not in seen:
                        seen.add(r)
                        if curriculum_phase == "errors_only" and r not in error_row_set:
                            continue
                        if curriculum_phase == "mix":
                            if r not in error_row_set and _random.random() > mix_ratio:
                                continue
                        candidate_rows.append((r, conflict_counts.get(r, 0), fd, cg))
            candidate_rows.sort(key=lambda x: -x[1])

            for row_idx, _, target_fd, target_cg in candidate_rows:
                if repair_round >= max_repair_rounds:
                    break
                if row_idx in processed_rows:
                    continue
                processed_rows.add(row_idx)

                # 使用初始 CG（不重建缓存），让所有候选行都经过模型决策
                # 正确的少数行也会被暴露给模型，模型需要通过特征判断该不该修
                current_cg = target_cg

                neighborhood = get_neighborhood(target_fd, fd_graph)

                env = RowLockRepairEnv(
                    target_row=row_idx,
                    initial_cg=current_cg,
                    dirty_df=current_df,
                    feature_extractor=self.fe,
                    neighborhood_fds=neighborhood,
                    clean_df=self.clean_df,  # 伪数据作为正确性参考（非真实GT）
                    reward_config=self.reward_config,
                    max_steps=ROW_LOCK_CONFIG["max_steps_per_episode"],
                    all_fds=self.fds,
                    precomputed_conflict_counts=conflict_counts,
                )

                obs, _ = env.reset()
                ep_reward, ep_len = 0.0, 0
                done = truncated = False
                mid_update = False

                # 收集数据时不需要梯度
                with torch.no_grad():
                    while not (done or truncated):
                        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        mask = env.get_action_mask()
                        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

                        dist, value = self.model(obs_t, mask_t)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                        next_obs, reward, done, truncated, info = env.step(action.item())

                        self.buffer.add(
                            obs=obs, action=action.item(), log_prob=log_prob.item(),
                            reward=reward, done=done or truncated,
                            value=value.item(), action_mask=mask,
                        )

                        obs = next_obs
                        ep_reward += reward
                        ep_len += 1
                        last_obs = obs
                        action_counts[action.item()] += 1

                        if self.buffer.full:
                            with torch.no_grad():
                                obs_t2 = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                                lv = self._raw_model.get_value(obs_t2).item()
                            self.buffer.compute_gae_returns(
                                last_value=lv,
                                gamma=self.cfg["ppo_gamma"],
                                lam=self.cfg["ppo_lam"],
                            )
                            mid_update = True
                            break

                if mid_update:
                    self.model.train()
                    self._ppo_update()
                    self.model.eval()
                    self.buffer.reset()
                    mid_update = False

                current_df = env.current_df
                if env.modified_cells:
                    cg_dirty = True
                repair_round += 1

                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_info = compute_episode_metrics(
                    initial_cg_violations=count_fd_violations(
                        epoch_dirty_df, target_fd
                    ),
                    final_cg_violations=count_fd_violations(current_df, target_fd),
                    modified_cells=env.modified_cells,
                    clean_df=self.clean_df,  # 伪数据作为正确性参考（非真实GT）
                    repaired_df=current_df,
                    original_dirty_df=epoch_dirty_df,
                )
                ep_w2r.append(ep_info.get("wrong2right", 0))
                ep_r2w.append(ep_info.get("right2wrong", 0))

            # buffer中还有剩余数据，做最后一次更新
            if self.buffer.size > 0:
                with torch.no_grad():
                    if last_obs is not None:
                        obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
                        last_value = self._raw_model.get_value(obs_t).item()
                self.buffer.compute_gae_returns(
                    last_value=last_value,
                    gamma=self.cfg["ppo_gamma"],
                    lam=self.cfg["ppo_lam"],
                )
                self.model.train()
                policy_losses, value_losses, entropies = self._ppo_update()
                self.model.eval()
            else:
                policy_losses, value_losses, entropies = [0.0], [0.0], [0.0]

            epoch_metrics = {
                "epoch": epoch + 1,
                "error_rate": error_rate,
                "mean_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
                "mean_ep_length": np.mean(ep_lengths) if ep_lengths else 0.0,
                "policy_loss": np.mean(policy_losses),
                "value_loss": np.mean(value_losses),
                "entropy": np.mean(entropies),
                "wrong2right": int(np.sum(ep_w2r)),
                "right2wrong": int(np.sum(ep_r2w)),
                "repair_rounds": repair_round,
                "action_counts": dict(action_counts),
                "elapsed": time.time() - t0,
            }
            self.metrics_history.append(epoch_metrics)

            # 使用 F1 选择最佳模型（基于伪数据计算的 W2R/R2W）
            w2r = epoch_metrics["wrong2right"]
            r2w = epoch_metrics["right2wrong"]
            precision = w2r / (w2r + r2w) if (w2r + r2w) > 0 else 0.0
            recall = w2r / (w2r + r2w) if (w2r + r2w) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                best_model_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics_history": self.metrics_history,
                    "f1_score": f1,
                }

            # entropy 退火
            if self.cfg.get("ppo_entropy_anneal", False):
                anneal_rate = self.cfg.get("ppo_entropy_anneal_rate", 0.98)
                min_ent = self.cfg.get("ppo_min_entropy", 0.0)
                self._current_entropy_coef = max(
                    self._current_entropy_coef * anneal_rate, min_ent
                )

            if (epoch + 1) % self.cfg["log_interval"] == 0:
                rhs_total = sum(action_counts[i] for i in range(N_RHS_ACTIONS))
                lhs_total = sum(action_counts[i] for i in range(N_RHS_ACTIONS, N_RHS_ACTIONS + N_LHS_ACTIONS))
                noop_total = action_counts[TOTAL_ACTIONS - 1]
                total_actions = sum(action_counts.values()) or 1
                print(
                    f"[PPO] Epoch {epoch+1:3d}/{self.cfg['ppo_epochs']} (err={error_rate:.2f} | {curriculum_phase}) | "
                    f"MeanRew: {epoch_metrics['mean_reward']:+.3f} | "
                    f"PolicyLoss: {epoch_metrics['policy_loss']:.4f} | "
                    f"Entropy: {epoch_metrics['entropy']:.4f} | "
                    f"W2R: {epoch_metrics['wrong2right']} R2W: {epoch_metrics['right2wrong']} | "
                    f"Rounds: {epoch_metrics['repair_rounds']} | "
                    f"Actions: RHS={rhs_total}({rhs_total/total_actions:.0%}) "
                    f"LHS={lhs_total}({lhs_total/total_actions:.0%}) "
                    f"NOOP={noop_total}({noop_total/total_actions:.0%}) | "
                    f"Time: {epoch_metrics['elapsed']:.1f}s"
                )

            if (epoch + 1) % self.cfg["save_interval"] == 0:
                self._save_checkpoint(epoch + 1)

        print("[PPO] Training complete.")

        # 返回最好的模型状态和指标历史
        return self.metrics_history, best_model_state, best_epoch

    def _ppo_update(self) -> Tuple[List[float], List[float], List[float]]:
        self.model.train()
        tensors = self.buffer.get_tensors(self.device)
        obs = tensors["obs"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]
        advantages = tensors["advantages"]
        returns = tensors["returns"]
        action_masks = tensors["action_masks"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies = [], [], []
        n = obs.shape[0]
        mini_batch_size = min(self.cfg["ppo_mini_batch_size"], n)

        # Save a clean copy of weights for NaN recovery
        backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        for _ in range(self.cfg["ppo_update_iters"]):
            indices = torch.randperm(n)
            for start in range(0, n, mini_batch_size):
                end = min(start + mini_batch_size, n)
                idx = indices[start:end]

                new_log_probs, entropy, values = self._raw_model.evaluate_actions(
                    obs[idx], actions[idx], action_masks[idx]
                )

                # NaN guard: skip this update if log_probs are NaN
                if torch.isnan(new_log_probs).any():
                    self.model.load_state_dict(backup_state)
                    self.model.eval()
                    return policy_losses or [0.0], value_losses or [0.0], entropies or [0.0]

                ratio = torch.exp(new_log_probs - old_log_probs[idx].detach())
                clip_eps = self.cfg["ppo_clip_epsilon"]
                surr1 = ratio * advantages[idx].detach()
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx].detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns[idx].detach())
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.cfg["ppo_vf_coef"] * value_loss
                    + self._current_entropy_coef * entropy_loss
                )

                # NaN guard: skip backward if loss is NaN
                if torch.isnan(total_loss):
                    self.model.load_state_dict(backup_state)
                    self.model.eval()
                    return policy_losses or [0.0], value_losses or [0.0], entropies or [0.0]

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["ppo_max_grad_norm"])
                self.optimizer.step()

                # NaN guard: check weights after step, restore if NaN appeared
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        self.model.load_state_dict(backup_state)
                        self.model.eval()
                        return policy_losses or [0.0], value_losses or [0.0], entropies or [0.0]

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return policy_losses, value_losses, entropies

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(self.cfg["checkpoint_dir"], f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }, path)
        print(f"[PPO] Checkpoint saved: {path}")
