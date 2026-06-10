"""
RL环境模块（gymnasium风格）
==========================
每个episode处理一个冲突组。

状态空间：
  特征提取器输出的固定维度向量（见features.py）

动作空间：
  离散动作，分为两类模板：
    - 动作 0 ~ N_RHS_ACTIONS-1：
        RHS统一动作，选第i个（按频率排序）右部候选值作为统一值
    - 动作 N_RHS_ACTIONS ~ N_RHS_ACTIONS + N_LHS_ACTIONS - 1：
        LHS移出动作，将第j行移到第k个左部替代候选
        （第一版简化：只选"移出哪行"，目标lhs固定选最近似组）
    - 动作 TOTAL_ACTIONS - 1：NO_OP（认为当前状态已最优）

终止条件：
  1. 冲突组无冲突（所有rhs候选值统一）
  2. 达到最大步数
  3. 选择了NO_OP

奖励设计：
  见config.py中的REWARD_CONFIG

简化假设：
  - 每次只处理一个冲突组（单FD，单group）
  - LHS移出时，目标lhs固定取cg.lhs_alternatives中频率最高的那个
  - 未来扩展：支持多FD联合处理
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from collections import Counter

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from .fd_utils import (
    ConflictGroup, FD,
    apply_rhs_unify, apply_lhs_eject,
    refresh_conflict_group, count_fd_violations,
    count_all_fd_violations, rank_lhs_alts_by_evidence,
    detect_row_in_conflict_groups,
)

from .features import FeatureExtractor, FEATURE_DIM, LHS_TOP_K
from config import REWARD_CONFIG, ENV_CONFIG


# 动作空间：RHS(3) + LHS(3) + NO_OP(1) = 7
N_RHS_ACTIONS = 3       # 选频率top3的RHS候选
N_LHS_ACTIONS = LHS_TOP_K  # 选top3 LHS候选之一
N_NO_OP = 1
TOTAL_ACTIONS = N_RHS_ACTIONS + N_LHS_ACTIONS + N_NO_OP  # 7


class FDRepairEnv(gym.Env):
    """
    FD冲突修复环境。

    初始化：
      env = FDRepairEnv(
          conflict_group=cg,
          dirty_df=dirty_df,
          clean_df=clean_df,        # 训练时提供，测试时为None
          feature_extractor=fe,
          reward_config=REWARD_CONFIG,
      )

    使用：
      obs, info = env.reset()
      obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        conflict_group: ConflictGroup,
        dirty_df: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        clean_df: Optional[pd.DataFrame] = None,
        reward_config: Optional[Dict] = None,
        max_steps: int = ENV_CONFIG["max_steps_per_episode"],
        all_fds: Optional[List[FD]] = None,
        initial_global_viol_total: Optional[int] = None,
    ):
        super().__init__()
        self.initial_cg = conflict_group
        self.initial_dirty_df = dirty_df.copy()
        self.clean_df = clean_df  # None表示测试模式（无GT）
        self.feature_extractor = feature_extractor
        self.reward_config = reward_config or REWARD_CONFIG
        self.max_steps = max_steps
        self.all_fds = all_fds or [conflict_group.fd]  # 全局FD列表

        # 是否有GT（训练模式）
        self.has_gt = clean_df is not None

        # 全局违反基准量：可由外部预算后传入（避免每次 episode 重复计算）
        if initial_global_viol_total is not None:
            self._initial_global_viol_total = initial_global_viol_total
        else:
            self._initial_global_viol_total = sum(
                count_all_fd_violations(self.initial_dirty_df, self.all_fds).values()
            )

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # 运行时状态（reset后初始化）
        self.current_df: pd.DataFrame = None
        self.current_cg: ConflictGroup = None
        self.step_count: int = 0
        self.modified_cells: Set[Tuple[int, str]] = set()
        self.action_history: List[Dict] = []
        self.initial_violations: int = 0
        self.initial_global_violations: int = 0

    # ──────────────────────────────────────────────────────────
    # 核心接口
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_df = self.initial_dirty_df.copy()
        self.current_cg = self.initial_cg
        self.step_count = 0
        self.modified_cells = set()
        self.action_history = []
        self.initial_global_violations = None

        # 算一次，后续 step 通过缓存取 prev_violations，_get_info 也复用
        self._cached_violations = count_fd_violations(self.current_df, self.current_cg.fd)
        self.initial_violations = self._cached_violations

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.current_df is not None, "Call reset() first."
        assert 0 <= action < TOTAL_ACTIONS, f"Invalid action {action}"

        self.step_count += 1
        # 直接使用缓存，避免重复调用 count_fd_violations
        prev_violations = self._cached_violations
        prev_global_violations = None

        # ── 解析并执行动作 ──
        action_info = self._decode_action(action)
        reward, done, action_info = self._execute_action(action, action_info, prev_violations, prev_global_violations)

        # ── 更新状态 ──
        self.current_cg = refresh_conflict_group(self.current_df, self.current_cg)
        self.action_history.append(action_info)

        # ── 终止判断 ──
        truncated = self.step_count >= self.max_steps
        if not done and truncated:
            # 超时终局奖励
            reward += self._terminal_reward()

        obs = self._get_obs()
        info = self._get_info()
        info["action_info"] = action_info

        return obs, reward, done, truncated, info

    # ──────────────────────────────────────────────────────────
    # 动作解码 & 执行
    # ──────────────────────────────────────────────────────────

    def _decode_action(self, action: int) -> Dict:
        """将整数action解码为结构化描述"""
        if action < N_RHS_ACTIONS:
            # 动作0：选频率最高的RHS候选（max-cor）
            return {
                "type": "rhs_unify",
                "candidate_index": 0,
            }
        elif action < N_RHS_ACTIONS + N_LHS_ACTIONS:
            lhs_idx = action - N_RHS_ACTIONS  # 0,1,2 对应top3候选
            return {
                "type": "lhs_eject",
                "lhs_alt_index": lhs_idx,
            }
        else:
            return {"type": "no_op"}

    def _execute_action(
        self, action: int, action_info: Dict, prev_violations: int, prev_global_violations: int
    ) -> Tuple[float, bool, Dict]:
        """
        执行动作，计算reward，判断done。

        返回：(reward, done, action_info_updated)
        """
        rc = self.reward_config
        reward = rc["step_cost"]  # 基础步骤成本

        # 处理列名大小写映射
        col_map = {c.lower(): c for c in self.current_df.columns}

        def get_mapped_col(col_name):
            """获取映射后的列名"""
            if col_name in self.current_df.columns:
                return col_name
            return col_map.get(col_name.lower(), col_name)

        if action_info["type"] == "no_op":
            # 无操作
            if self.current_cg.has_conflict:
                reward += rc["no_op_penalty"]
                done = False  # 还有冲突但选了无操作，继续（允许再尝试）
            else:
                reward += self._terminal_reward()
                done = True
            action_info["executed"] = True
            return reward, done, action_info

        elif action_info["type"] == "rhs_unify":
            cand_idx = action_info["candidate_index"]
            rhs_list = self.current_cg.rhs_candidate_list()

            if cand_idx >= len(rhs_list):
                # 无效动作（候选值不够）
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = "invalid: candidate_index out of range"
                return reward, False, action_info

            target_val = rhs_list[cand_idx]
            action_info["target_value"] = target_val

            # 获取映射后的列名
            rhs_col = get_mapped_col(self.current_cg.fd.rhs_col)

            # 记录修改前的状态（用于reward计算）
            cells_before = {}
            for idx in self.current_cg.row_indices:
                cells_before[idx] = self.current_df.at[idx, rhs_col]

            # 执行修改
            self.current_df = apply_rhs_unify(
                self.current_df, self.current_cg, target_val
            )

            # 记录被修改的单元格
            for idx in self.current_cg.row_indices:
                old_val = cells_before[idx]
                new_val = self.current_df.at[idx, rhs_col]
                if old_val != new_val:
                    self.modified_cells.add((idx, rhs_col))

            # 计算正确性reward
            correctness_reward = self._compute_correctness_reward(cells_before, rhs_col, target_val)
            reward += correctness_reward

            # 计算局部FD冲突变化reward
            new_violations = count_fd_violations(self.current_df, self.current_cg.fd)
            self._cached_violations = new_violations   # 更新缓存供下一步使用
            delta = prev_violations - new_violations
            reward += delta * rc["conflict_reduce"] if delta > 0 else abs(delta) * rc["conflict_increase"]

            # 终止判断
            done = not refresh_conflict_group(self.current_df, self.current_cg).has_conflict
            global_delta = delta
            if global_delta > 0:
                reward += global_delta * rc["global_viol_reduce"]
            elif global_delta < 0:
                reward += abs(global_delta) * rc["global_viol_increase"]
            if done:
                reward += self._terminal_reward()

            action_info["executed"] = True
            action_info["correctness_reward"] = correctness_reward
            action_info["global_delta"] = global_delta
            return reward, done, action_info

        elif action_info["type"] == "lhs_eject":
            lhs_idx = action_info["lhs_alt_index"]

            # 先确定 eject_row（少数派行），再用证据列重排序候选集
            majority_rhs = self.current_cg.majority_rhs
            rhs_col = get_mapped_col(self.current_cg.fd.rhs_col)
            eject_candidates = [
                idx for idx in self.current_cg.row_indices
                if self.current_df.at[idx, rhs_col] != majority_rhs
            ]
            if not eject_candidates:
                eject_candidates = self.current_cg.row_indices[:1]
            eject_row = eject_candidates[0]

            # 用证据列对 LHS 候选集重排序
            lhs_alts_list = rank_lhs_alts_by_evidence(
                self.current_df,
                self.current_cg.fd,
                self.all_fds,
                self.current_cg.lhs_alternatives,
                eject_row,
            )

            if lhs_idx >= len(lhs_alts_list) or not lhs_alts_list:
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = "invalid: no lhs alternatives"
                return reward, False, action_info

            target_lhs, _, _ = lhs_alts_list[lhs_idx]
            action_info["target_lhs"] = target_lhs
            action_info["eject_row"] = eject_row

            # 记录修改前lhs值
            lhs_cols = [get_mapped_col(col) for col in self.current_cg.fd.lhs_cols]
            cells_before = {
                (eject_row, col): self.current_df.at[eject_row, col]
                for col in lhs_cols
            }

            # 执行移出
            try:
                self.current_df = apply_lhs_eject(
                    self.current_df, self.current_cg, eject_row, target_lhs
                )
            except ValueError as e:
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = str(e)
                return reward, False, action_info

            # 记录修改
            for col in lhs_cols:
                old_val = cells_before[(eject_row, col)]
                new_val = self.current_df.at[eject_row, col]
                if old_val != new_val:
                    self.modified_cells.add((eject_row, col))

            # 计算正确性reward（左部修复）
            correctness_reward = 0.0
            if self.has_gt:
                for col in lhs_cols:
                    old_dirty = cells_before[(eject_row, col)]
                    new_val = self.current_df.at[eject_row, col]
                    clean_val = self.clean_df.at[eject_row, col]
                    cell_r = self._cell_reward(old_dirty, new_val, clean_val)
                    correctness_reward += cell_r
            reward += correctness_reward

            # 局部FD冲突变化
            new_violations = count_fd_violations(self.current_df, self.current_cg.fd)
            self._cached_violations = new_violations
            delta = prev_violations - new_violations
            reward += delta * rc["conflict_reduce"] if delta > 0 else abs(delta) * rc["conflict_increase"]

            done = not refresh_conflict_group(self.current_df, self.current_cg).has_conflict
            global_delta = 0
            if done:
                new_global_total = sum(count_all_fd_violations(self.current_df, self.all_fds).values())
                global_delta = self._initial_global_viol_total - new_global_total
                if global_delta > 0:
                    reward += global_delta * rc["global_viol_reduce"]
                elif global_delta < 0:
                    reward += abs(global_delta) * rc["global_viol_increase"]
                reward += self._terminal_reward()

            action_info["executed"] = True
            action_info["correctness_reward"] = correctness_reward
            action_info["global_delta"] = global_delta
            return reward, done, action_info

        return reward, False, action_info

    # ──────────────────────────────────────────────────────────
    # Reward 辅助函数
    # ──────────────────────────────────────────────────────────

    def _compute_correctness_reward(
        self, cells_before: Dict[int, Any], rhs_col: str, target_val: Any
    ) -> float:
        """
        计算右部统一动作的正确性reward。

        只有在有GT（训练模式）时才计算；测试模式返回0。
        """
        if not self.has_gt:
            return 0.0

        rc = self.reward_config
        total = 0.0
        for idx, old_val in cells_before.items():
            new_val = target_val
            clean_val = self.clean_df.at[idx, rhs_col]
            total += self._cell_reward(old_val, new_val, clean_val)
        return total

    def _cell_reward(self, old_val: Any, new_val: Any, clean_val: Any) -> float:
        """
        单元格级别的reward计算：

          old_val  : 执行动作前的dirty值
          new_val  : 执行动作后的新值
          clean_val: ground truth值
        """
        rc = self.reward_config
        old_correct = (old_val == clean_val)
        new_correct = (new_val == clean_val)

        if not old_correct and new_correct:
            return rc["wrong_to_right"]      # 修复成功
        elif old_correct and not new_correct:
            # 改错了正确值！记录调试信息
            if hasattr(self, '_debug_r2w'):
                self._debug_r2w.append({
                    'has_conflict': self.current_cg.has_conflict,
                    'old_val': old_val,
                    'new_val': new_val,
                    'clean_val': clean_val,
                })
            return rc["right_to_wrong"]      # 破坏正确值
        elif not old_correct and not new_correct:
            return rc["wrong_to_wrong"]      # 错改错
        else:
            return rc["right_to_right"]

    def _get_correct_rhs_value(self, rhs_col: str):
        """
        从 clean_df 中查询当前 group 对应的正确 RHS 值。
        返回正确值，或 None（无法确定时）。
        """
        if not self.has_gt:
            return None
        col_map = {c.lower(): c for c in self.clean_df.columns}
        clean_rhs = rhs_col if rhs_col in self.clean_df.columns else col_map.get(rhs_col.lower())
        if not clean_rhs:
            return None

        mask = pd.Series([True] * len(self.clean_df), index=self.clean_df.index)
        lhs_cols = self.current_cg.fd.lhs_cols
        for col, val in zip(lhs_cols, self.current_cg.lhs_value):
            actual = col if col in self.clean_df.columns else col_map.get(col.lower(), col)
            mask &= (self.clean_df[actual] == val)

        matched = self.clean_df[mask]
        if len(matched) == 0:
            return None
        mode = matched[clean_rhs].mode()
        return mode[0] if len(mode) > 0 else None

    def _has_lhs_error(self) -> bool:
        """
        判断当前 group 是否包含 LHS 错误的行（dirty LHS ≠ clean LHS）。
        """
        if not self.has_gt:
            return False
        col_map_d = {c.lower(): c for c in self.current_df.columns}
        col_map_c = {c.lower(): c for c in self.clean_df.columns}
        for row_idx in self.current_cg.row_indices:
            for col in self.current_cg.fd.lhs_cols:
                dc = col if col in self.current_df.columns else col_map_d.get(col.lower(), col)
                cc = col if col in self.clean_df.columns else col_map_c.get(col.lower(), col)
                if dc in self.current_df.columns and cc in self.clean_df.columns:
                    if self.current_df.at[row_idx, dc] != self.clean_df.at[row_idx, cc]:
                        return True
        return False

    def _terminal_reward(self) -> float:
        """
        终局奖励：根据最终修复质量给奖励。

        训练模式：比较修改单元格与GT的一致性
        测试模式：只看是否消除了冲突
        """
        rc = self.reward_config

        if self.has_gt:
            # 计算被修改单元格的正确率
            correct = 0
            total = len(self.modified_cells)
            if total == 0:
                # 没有修改且无冲突：说明原来就对
                if not self.current_cg.has_conflict:
                    return rc["terminal_correct_bonus"] * 0.5
                return 0.0

            for (idx, col) in self.modified_cells:
                if self.current_df.at[idx, col] == self.clean_df.at[idx, col]:
                    correct += 1

            accuracy = correct / total
            if accuracy >= 0.9:
                return rc["terminal_correct_bonus"]
            elif accuracy >= 0.5:
                return rc["terminal_partial_bonus"]
            else:
                return rc["terminal_fail_penalty"]
        else:
            # 测试模式：只看冲突
            if not refresh_conflict_group(self.current_df, self.current_cg).has_conflict:
                return rc["terminal_correct_bonus"] * 0.3  # 保守奖励
            return rc["terminal_fail_penalty"] * 0.5

    # ──────────────────────────────────────────────────────────
    # 观测 & 信息
    # ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return self.feature_extractor.extract(
            self.current_cg, self.current_df,
            self.step_count, self.max_steps
        )

    def _get_info(self) -> Dict:
        return {
            "step": self.step_count,
            "has_conflict": self.current_cg.has_conflict,
            "n_conflicts": self._cached_violations,
            "modified_cells": len(self.modified_cells),
            "rhs_candidates": dict(self.current_cg.rhs_candidates),
        }

    def get_action_mask(self) -> np.ndarray:
        """
        返回合法动作的布尔掩码，形状 (TOTAL_ACTIONS,)。
        动作编号：0=RHS_UNIFY, 1/2/3=LHS_TOP3, 4=NO_OP
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

        # RHS动作：只有存在候选才开放
        if len(self.current_cg.rhs_candidates) > 0:
            mask[0] = True

        # LHS动作：按实际候选数量开放top3
        col_map_cur = {c.lower(): c for c in self.current_df.columns}
        rhs_col = self.current_cg.fd.rhs_col if self.current_cg.fd.rhs_col in self.current_df.columns \
            else col_map_cur.get(self.current_cg.fd.rhs_col.lower())
        rows = self.current_cg.row_indices
        if rhs_col and self.current_cg.rhs_candidates:
            majority_rhs = max(self.current_cg.rhs_candidates, key=self.current_cg.rhs_candidates.get)
            minority_rows = [r for r in rows if self.current_df.at[r, rhs_col] != majority_rhs]
            eject_row = minority_rows[0] if minority_rows else (rows[0] if rows else 0)
        else:
            eject_row = rows[0] if rows else 0

        lhs_alts_list = rank_lhs_alts_by_evidence(
            self.current_df, self.current_cg.fd, self.all_fds,
            self.current_cg.lhs_alternatives, eject_row,
        )
        for i in range(min(len(lhs_alts_list), N_LHS_ACTIONS)):
            mask[N_RHS_ACTIONS + i] = True

        mask[-1] = True  # NO_OP always available

        return mask

    def render(self, mode="human"):
        print(f"Step {self.step_count} | CG: {self.current_cg}")
        print(f"  RHS candidates: {self.current_cg.rhs_candidates}")
        print(f"  Has conflict: {self.current_cg.has_conflict}")


class RowLockRepairEnv(gym.Env):
    """
    行锁级联修复环境。

    Episode = 1 个 minority row 的追踪修复。
    锁住该 row，逐 step 在邻域 FD 间级联修复，直到该 row 在所有 FD 都不再是 minority。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_row: int,
        initial_cg: ConflictGroup,
        dirty_df: pd.DataFrame,
        feature_extractor,
        neighborhood_fds: List[FD],
        clean_df: Optional[pd.DataFrame] = None,
        reward_config: Optional[Dict] = None,
        max_steps: int = 20,
        all_fds: Optional[List[FD]] = None,
        use_lite: bool = False,
        precomputed_conflict_counts: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.target_row = target_row
        self.initial_cg = initial_cg
        self.initial_dirty_df = dirty_df.copy()
        self.feature_extractor = feature_extractor
        self.neighborhood_fds = neighborhood_fds
        self.clean_df = clean_df
        self.reward_config = reward_config or REWARD_CONFIG
        self.max_steps = max_steps
        self.all_fds = all_fds or [initial_cg.fd]
        self.use_lite = use_lite
        self._precomputed_conflict_counts = precomputed_conflict_counts

        self.has_gt = clean_df is not None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Runtime state
        self.current_df: pd.DataFrame = None
        self.current_cg: ConflictGroup = None
        self.step_count: int = 0
        self.modified_cells: Set[Tuple[int, str]] = set()
        self.action_history: List[Dict] = []
        self.cascade_depth: int = 0
        self._cached_lhs_alts: Optional[List] = None  # 共享给 get_action_mask
        self._cached_eject_row: Optional[int] = None
        self._cached_conflict_counts: Dict[int, int] = {}  # reset 时计算
        self._cached_cec: Dict[int, float] = {}  # reset 时计算
        self._lite: bool = False  # 两阶段：reset 时 lite=True，首次修复动作后切为 False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_df = self.initial_dirty_df.copy()
        self.current_cg = self.initial_cg
        self.step_count = 0
        self.modified_cells = set()
        self.action_history = []
        self.cascade_depth = 0
        self._cached_violations = count_fd_violations(self.current_df, self.current_cg.fd)
        self._cached_lhs_alts = None  # lite 阶段不计算，延迟到首次修复动作后
        # 冲突计数：优先使用外部预计算值（trainer 缓存），否则自行计算
        if self._precomputed_conflict_counts is not None:
            self._cached_conflict_counts = self._precomputed_conflict_counts
        else:
            from .fd_utils import compute_row_conflict_counts as _crcc
            self._cached_conflict_counts = _crcc(self.current_df, self.all_fds)
        # CEC：按需计算（首次调用 _get_obs 时懒计算，多数 episode 第一步就是 NO_OP 无需 CEC）
        self._cached_cec: Dict[int, float] = {}
        # 两阶段：仅 use_lite=True 时用 lite 特征（跳过 EVIDENCE+LHS）
        self._lite = self.use_lite

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.current_df is not None, "Call reset() first."
        assert 0 <= action < TOTAL_ACTIONS, f"Invalid action {action}"

        self.step_count += 1
        prev_violations = self._cached_violations

        # Parse and execute action
        action_info = self._decode_action(action)
        reward, done, action_info = self._execute_action(action, action_info, prev_violations)

        # Update current CG
        self.current_cg = refresh_conflict_group(self.current_df, self.current_cg)
        self.action_history.append(action_info)

        # Row-lock: scan target_row minority status across neighborhood FDs
        if not done:
            minority_cgs = detect_row_in_conflict_groups(
                self.target_row, self.neighborhood_fds, self.current_df
            )

            if not minority_cgs:
                # Row is clean across all neighborhood FDs -> natural end
                done = True
                reward += self._terminal_reward()
            else:
                # Row is still minority in some CGs -> switch to highest-priority CG
                next_fd, next_cg = minority_cgs[0]
                if next_cg != self.current_cg:
                    self.cascade_depth += 1
                    self.current_cg = next_cg
                    self._cached_violations = count_fd_violations(
                        self.current_df, next_cg.fd
                    )
                else:
                    self._cached_violations = count_fd_violations(
                        self.current_df, self.current_cg.fd
                    )

        # Termination check
        truncated = self.step_count >= self.max_steps
        if not done and truncated:
            reward += self._terminal_reward()

        # 首次非 NO_OP 动作后切换为 full 特征，并清除 CEC 缓存（数据已变）
        if action_info["type"] != "no_op":
            self._lite = False
            self._cached_cec.pop(self.target_row, None)

        obs = self._get_obs()
        info = self._get_info()
        info["action_info"] = action_info

        return obs, reward, done, truncated, info

    # Action decoding
    def _decode_action(self, action: int) -> Dict:
        """Decode integer action to structured description (same as FDRepairEnv)."""
        if action < N_RHS_ACTIONS:
            return {"type": "rhs_unify", "candidate_index": action}
        elif action < N_RHS_ACTIONS + N_LHS_ACTIONS:
            return {"type": "lhs_eject", "lhs_alt_index": action - N_RHS_ACTIONS}
        else:
            return {"type": "no_op"}

    # Action execution
    def _execute_action(self, action, action_info, prev_violations):
        """Execute action, compute reward. Similar to FDRepairEnv but with NO_OP checking row minority."""
        rc = self.reward_config
        reward = rc["step_cost"]
        done = False

        col_map = {c.lower(): c for c in self.current_df.columns}

        def get_mapped_col(col_name):
            if col_name in self.current_df.columns:
                return col_name
            return col_map.get(col_name.lower(), col_name)

        if action_info["type"] == "no_op":
            # NO_OP 始终允许执行，由 terminal reward 的 selection 奖励判断对错
            reward += self._terminal_reward()
            done = True
            action_info["executed"] = True
            return reward, done, action_info

        elif action_info["type"] == "rhs_unify":
            cand_idx = action_info["candidate_index"]
            rhs_list = self.current_cg.rhs_candidate_list()

            if cand_idx >= len(rhs_list):
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = "invalid: candidate_index out of range"
                return reward, False, action_info

            target_val = rhs_list[cand_idx]
            action_info["target_value"] = target_val

            rhs_col = get_mapped_col(self.current_cg.fd.rhs_col)

            cells_before = {}
            for idx in self.current_cg.row_indices:
                cells_before[idx] = self.current_df.at[idx, rhs_col]

            self.current_df = apply_rhs_unify(
                self.current_df, self.current_cg, target_val
            )

            for idx in self.current_cg.row_indices:
                old_val = cells_before[idx]
                new_val = self.current_df.at[idx, rhs_col]
                if old_val != new_val:
                    self.modified_cells.add((idx, rhs_col))

            correctness_reward = self._compute_correctness_reward(
                cells_before, rhs_col, target_val
            )
            reward += correctness_reward

            new_violations = count_fd_violations(self.current_df, self.current_cg.fd)
            self._cached_violations = new_violations
            delta = prev_violations - new_violations
            reward += delta * rc["conflict_reduce"] if delta > 0 else abs(delta) * rc["conflict_increase"]

            if done:
                reward += self._terminal_reward()

            action_info["executed"] = True
            action_info["correctness_reward"] = correctness_reward
            return reward, done, action_info

        elif action_info["type"] == "lhs_eject":
            lhs_idx = action_info["lhs_alt_index"]

            majority_rhs = self.current_cg.majority_rhs
            rhs_col = get_mapped_col(self.current_cg.fd.rhs_col)
            eject_candidates = [
                idx for idx in self.current_cg.row_indices
                if self.current_df.at[idx, rhs_col] != majority_rhs
            ]
            if not eject_candidates:
                eject_candidates = self.current_cg.row_indices[:1]
            eject_row = eject_candidates[0]

            lhs_alts_list = rank_lhs_alts_by_evidence(
                self.current_df, self.current_cg.fd, self.all_fds,
                self.current_cg.lhs_alternatives, eject_row,
            )

            if lhs_idx >= len(lhs_alts_list) or not lhs_alts_list:
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = "invalid: no lhs alternatives"
                return reward, False, action_info

            target_lhs, _, _ = lhs_alts_list[lhs_idx]
            action_info["target_lhs"] = target_lhs
            action_info["eject_row"] = eject_row

            lhs_cols = [get_mapped_col(col) for col in self.current_cg.fd.lhs_cols]
            cells_before = {
                (eject_row, col): self.current_df.at[eject_row, col]
                for col in lhs_cols
            }

            try:
                self.current_df = apply_lhs_eject(
                    self.current_df, self.current_cg, eject_row, target_lhs
                )
            except ValueError as e:
                reward += rc["invalid_action"]
                action_info["executed"] = False
                action_info["reason"] = str(e)
                return reward, False, action_info

            for col in lhs_cols:
                old_val = cells_before[(eject_row, col)]
                new_val = self.current_df.at[eject_row, col]
                if old_val != new_val:
                    self.modified_cells.add((eject_row, col))

            correctness_reward = 0.0
            if self.has_gt:
                for col in lhs_cols:
                    old_dirty = cells_before[(eject_row, col)]
                    new_val = self.current_df.at[eject_row, col]
                    clean_val = self.clean_df.at[eject_row, col]
                    cell_r = self._cell_reward(old_dirty, new_val, clean_val)
                    correctness_reward += cell_r
            reward += correctness_reward

            new_violations = count_fd_violations(self.current_df, self.current_cg.fd)
            self._cached_violations = new_violations
            delta = prev_violations - new_violations
            reward += delta * rc["conflict_reduce"] if delta > 0 else abs(delta) * rc["conflict_increase"]

            if done:
                reward += self._terminal_reward()

            action_info["executed"] = True
            action_info["correctness_reward"] = correctness_reward
            return reward, done, action_info

        return reward, False, action_info

    # Reward helpers
    def _compute_correctness_reward(self, cells_before, rhs_col, target_val):
        if not self.has_gt:
            return 0.0
        rc = self.reward_config
        total = 0.0
        for idx, old_val in cells_before.items():
            new_val = target_val
            clean_val = self.clean_df.at[idx, rhs_col]
            total += self._cell_reward(old_val, new_val, clean_val)
        return total

    def _cell_reward(self, old_val, new_val, clean_val):
        rc = self.reward_config
        old_correct = (old_val == clean_val)
        new_correct = (new_val == clean_val)
        if not old_correct and new_correct:
            return rc["wrong_to_right"]
        elif old_correct and not new_correct:
            return rc["right_to_wrong"]
        elif not old_correct and not new_correct:
            return rc["wrong_to_wrong"]
        else:
            return rc["right_to_right"]

    def _terminal_reward(self):
        rc = self.reward_config
        selection_reward = 0.0

        # ── 行选择奖励：基于原始状态判断"该不该修" ──
        if self.has_gt:
            fd = self.initial_cg.fd
            rhs_col = fd.rhs_col
            if rhs_col in self.initial_dirty_df.columns and rhs_col in self.clean_df.columns:
                try:
                    original_rhs = str(self.initial_dirty_df.at[self.target_row, rhs_col])
                    clean_rhs = str(self.clean_df.at[self.target_row, rhs_col])
                    needs_repair = (original_rhs != clean_rhs)
                    modified = len(self.modified_cells) > 0

                    if modified and needs_repair:
                        selection_reward = rc["correct_selection_bonus"]
                    elif modified and not needs_repair:
                        selection_reward = rc["wrong_selection_penalty"]
                    elif not modified and needs_repair:
                        selection_reward = rc["missed_repair_penalty"]
                    else:  # not modified and not needs_repair
                        selection_reward = rc["correct_skip_bonus"]
                except Exception:
                    pass

        if self.has_gt:
            total = len(self.modified_cells)
            if total == 0:
                if not self.current_cg.has_conflict:
                    return rc["terminal_correct_bonus"] * 0.5 + selection_reward
                return selection_reward
            correct = 0
            for (idx, col) in self.modified_cells:
                if self.current_df.at[idx, col] == self.clean_df.at[idx, col]:
                    correct += 1
            accuracy = correct / total
            if accuracy >= 0.9:
                base = rc["terminal_correct_bonus"]
            elif accuracy >= 0.5:
                base = rc["terminal_partial_bonus"]
            else:
                base = rc["terminal_fail_penalty"]
            return base + selection_reward
        else:
            minority_cgs = detect_row_in_conflict_groups(
                self.target_row, self.neighborhood_fds, self.current_df
            )
            if not minority_cgs:
                return rc["terminal_correct_bonus"] * 0.3
            return rc["terminal_fail_penalty"] * 0.5

    # Observation
    def _get_obs(self) -> np.ndarray:
        minority_cgs = detect_row_in_conflict_groups(
            self.target_row, self.neighborhood_fds, self.current_df
        )
        affected_fd_count = len(set(fd for fd, _ in minority_cgs))
        row_lhs_degree = len(self.neighborhood_fds)

        # 目标行的全局冲突总数（reset 时计算或外部传入，缓存复用）
        total_conflict_count = float(self._cached_conflict_counts.get(self.target_row, 0))

        # 条件证据一致性：懒计算（首次调用时计算并缓存）
        if self.target_row not in self._cached_cec:
            from .fd_utils import compute_conditional_evidence_consistency as _cec
            self._cached_cec[self.target_row] = _cec(
                self.current_df, self.target_row,
                self.all_fds, self.all_fds, self.neighborhood_fds)
        conditional_evidence_consistency = float(
            self._cached_cec.get(self.target_row, 0.5))

        # 预计算 LHS 候选列表（缓存给 get_action_mask 复用）
        # lite 模式：跳过昂贵的 rank_lhs_alts_by_evidence，mask 用简化逻辑
        if not self._lite:
            col_map_cur = {c.lower(): c for c in self.current_df.columns}
            rhs_col_name = self.current_cg.fd.rhs_col if self.current_cg.fd.rhs_col in self.current_df.columns \
                else col_map_cur.get(self.current_cg.fd.rhs_col.lower())
            rows = self.current_cg.row_indices
            if rhs_col_name and self.current_cg.rhs_candidates:
                majority_rhs = max(self.current_cg.rhs_candidates, key=self.current_cg.rhs_candidates.get)
                minority_rows = [r for r in rows if self.current_df.at[r, rhs_col_name] != majority_rhs]
                eject_row = minority_rows[0] if minority_rows else (rows[0] if rows else 0)
            else:
                eject_row = rows[0] if rows else 0

            self._cached_lhs_alts = rank_lhs_alts_by_evidence(
                self.current_df, self.current_cg.fd, self.all_fds,
                self.current_cg.lhs_alternatives, eject_row,
            )
            self._cached_eject_row = eject_row
        else:
            # lite 阶段不计算 LHS 排名，mask 用简化版
            self._cached_lhs_alts = None

        return self.feature_extractor.extract(
            self.current_cg, self.current_df,
            self.step_count, self.max_steps,
            lock_step_count=self.step_count,
            affected_fd_count=affected_fd_count,
            row_lhs_degree=row_lhs_degree,
            cascade_depth=self.cascade_depth,
            total_conflict_count=total_conflict_count,
            conditional_evidence_consistency=conditional_evidence_consistency,
            lhs_alts_list=self._cached_lhs_alts,
            lite=self._lite,
        )

    def _get_info(self) -> Dict:
        return {
            "step": self.step_count,
            "has_conflict": self.current_cg.has_conflict,
            "modified_cells": len(self.modified_cells),
            "target_row": self.target_row,
            "cascade_depth": self.cascade_depth,
        }

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)

        if len(self.current_cg.rhs_candidates) > 0:
            for i in range(min(len(self.current_cg.rhs_candidate_list()), N_RHS_ACTIONS)):
                mask[i] = True

        # 复用 _get_obs() 中缓存的 LHS 候选列表；lite 模式用简化逻辑
        lhs_alts_list = self._cached_lhs_alts
        if lhs_alts_list is None and not self._lite:
            # 兜底：如果 _get_obs 没被调用过，自己计算
            col_map_cur = {c.lower(): c for c in self.current_df.columns}
            rhs_col = self.current_cg.fd.rhs_col if self.current_cg.fd.rhs_col in self.current_df.columns \
                else col_map_cur.get(self.current_cg.fd.rhs_col.lower())
            rows = self.current_cg.row_indices
            if rhs_col and self.current_cg.rhs_candidates:
                majority_rhs = max(self.current_cg.rhs_candidates, key=self.current_cg.rhs_candidates.get)
                minority_rows = [r for r in rows if self.current_df.at[r, rhs_col] != majority_rhs]
                eject_row = minority_rows[0] if minority_rows else (rows[0] if rows else 0)
            else:
                eject_row = rows[0] if rows else 0
            lhs_alts_list = rank_lhs_alts_by_evidence(
                self.current_df, self.current_cg.fd, self.all_fds,
                self.current_cg.lhs_alternatives, eject_row,
            )

        if lhs_alts_list is not None:
            for i in range(min(len(lhs_alts_list), N_LHS_ACTIONS)):
                mask[N_RHS_ACTIONS + i] = True
        elif self._lite:
            # lite 模式：按 lhs_alternatives 数量开放 LHS 动作（不排序，省去 rank 开销）
            n_lhs = len(self.current_cg.lhs_alternatives) if self.current_cg.lhs_alternatives else 0
            for i in range(min(n_lhs, N_LHS_ACTIONS)):
                mask[N_RHS_ACTIONS + i] = True

        mask[-1] = True  # NO_OP always available

        return mask

    def render(self, mode="human"):
        print(f"Step {self.step_count} | Row: {self.target_row} | "
              f"CG: {self.current_cg.fd} lhs={self.current_cg.lhs_value} | "
              f"Cascade: {self.cascade_depth}")
