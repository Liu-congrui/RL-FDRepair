"""
策略模型（PPO Actor-Critic）
============================
简化版：去除错误类型分类器，直接对动作打分。

动作空间：RHS(N_RHS_ACTIONS) + LHS(LHS_TOP_K) + NO_OP(1)
特征维度：GROUP(8) + RHS(4) + EVIDENCE(4) + LHS(LHS_TOP_K*5) = 31（+ ROW_LOCK 4 + TARGET_ROW 2 = 37）

网络结构：
  输入(37) -> 共享context编码器 -> [Actor头, Critic头]
  Actor: 对RHS/LHS各候选独立打分，concat后输出N_RHS+LHS_TOP_K+1个logit
  Critic: 输出状态价值标量
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .features import (
    FEATURE_DIM, GROUP_FEATURES, EVIDENCE_FEATURES,
    RHS_FEATURES, LHS_CANDIDATE_FEATURES, LHS_TOP_K,
)
from .environment import TOTAL_ACTIONS, N_RHS_ACTIONS, N_LHS_ACTIONS
from config import NETWORK_CONFIG


# 特征向量切片位置（与 features.py 的 extend 顺序一致）
# features.py 顺序：GROUP(8) + RHS(4) + EVIDENCE(4) + LHS(15) = 31
_GROUP_START = 0
_GROUP_END = GROUP_FEATURES                                         # 8
_RHS_START = _GROUP_END                                             # 8
_RHS_END = _RHS_START + RHS_FEATURES                                # 12
_EVIDENCE_START = _RHS_END                                          # 12
_EVIDENCE_END = _EVIDENCE_START + EVIDENCE_FEATURES                 # 16
_LHS_START = _EVIDENCE_END                                          # 16
_LHS_END = _LHS_START + LHS_TOP_K * LHS_CANDIDATE_FEATURES          # 31


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(),
    )


class ActorCritic(nn.Module):
    """
    Actor-Critic网络。

    RHS动作(1个)：直接用context+rhs全局特征打分
    LHS动作(3个)：每个候选独立编码后与context一起打分（共享权重）
    NO_OP(1个)：只依赖context
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = NETWORK_CONFIG["hidden_dim"],
        feature_dim: int = NETWORK_CONFIG["feature_dim"],
        n_actions: int = TOTAL_ACTIONS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # context编码器：全量特征（含 row-lock features） → embedding
        self.context_encoder = _make_mlp(input_dim, hidden_dim, feature_dim)

        # RHS打分：concat(ctx_emb, evidence_feat, rhs_global_feat) → N_RHS_ACTIONS
        self.rhs_scorer = nn.Sequential(
            nn.Linear(feature_dim + EVIDENCE_FEATURES + RHS_FEATURES, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_RHS_ACTIONS),  # top-3 RHS candidates
        )

        # LHS候选编码器（共享权重，消除位置偏置）
        self.lhs_encoder = _make_mlp(LHS_CANDIDATE_FEATURES, hidden_dim // 2, feature_dim // 2)

        # LHS打分：concat(ctx_emb, lhs_cand_emb) → scalar
        self.lhs_scorer = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # NO_OP打分：只依赖context
        self.noop_scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Critic
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        for scorer in (self.rhs_scorer, self.lhs_scorer, self.noop_scorer):
            nn.init.orthogonal_(scorer[-1].weight, gain=0.01)

    def _compute_logits(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]

        # NaN guard: replace NaN in input features with 0
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        ctx_raw = obs[:, _GROUP_START:_GROUP_END]           # (B, GROUP_FEATURES)
        rhs_raw = obs[:, _RHS_START:_RHS_END]               # (B, RHS_FEATURES)
        evidence_raw = obs[:, _EVIDENCE_START:_EVIDENCE_END]  # (B, EVIDENCE_FEATURES)
        lhs_raw = obs[:, _LHS_START:_LHS_END]               # (B, LHS_TOP_K * LHS_CANDIDATE_FEATURES)

        # context编码
        ctx_emb = self.context_encoder(obs)  # (B, feature_dim) — full obs includes row-lock features

        # RHS动作打分（N_RHS_ACTIONS个）
        rhs_scores = self.rhs_scorer(
            torch.cat([ctx_emb, evidence_raw, rhs_raw], dim=-1)
        )  # (B, N_RHS_ACTIONS)

        # LHS候选打分（LHS_TOP_K个，共享权重）
        lhs_cands = lhs_raw.reshape(batch, LHS_TOP_K, LHS_CANDIDATE_FEATURES)
        lhs_flat = lhs_cands.reshape(batch * LHS_TOP_K, LHS_CANDIDATE_FEATURES)
        lhs_emb = self.lhs_encoder(lhs_flat).reshape(batch, LHS_TOP_K, self.feature_dim // 2)

        ctx_exp = ctx_emb.unsqueeze(1).expand(-1, LHS_TOP_K, -1)  # (B, LHS_TOP_K, feature_dim)
        lhs_scores = self.lhs_scorer(
            torch.cat([ctx_exp, lhs_emb], dim=-1)
        ).squeeze(-1)  # (B, LHS_TOP_K)

        # NO_OP打分（1个）
        noop_score = self.noop_scorer(ctx_emb)  # (B, 1)

        logits = torch.cat([rhs_scores, lhs_scores, noop_score], dim=-1)  # (B, N_RHS+LHS_TOP_K+1)

        # Clamp to safe range: exp(80) ≈ 5e34, just under float32 max ≈ 3.4e38
        # If logits can reach ±1e6, log_prob diff can be ~2e6, exp(2e6) = inf → NaN
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        return logits

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        logits = self._compute_logits(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
            # Safeguard: if all actions are masked out, allow NO_OP (last action)
            all_masked = torch.all(torch.isinf(logits), dim=-1, keepdim=True)
            logits = torch.where(all_masked,
                                torch.full_like(logits, float("-inf")),
                                logits)
            logits[:, -1] = torch.where(all_masked.squeeze(-1), 0.0, logits[:, -1])

        ctx_emb = self.context_encoder(obs)
        value = self.critic_head(ctx_emb)

        return Categorical(logits=logits), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        ctx_emb = self.context_encoder(obs)
        return self.critic_head(ctx_emb)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs, action_mask)
        return dist.log_prob(actions), dist.entropy(), values


# ============================================================
# PPO Rollout Buffer
# ============================================================

class RolloutBuffer:
    """存储PPO训练所需的rollout数据。"""

    def __init__(self, buffer_size: int, obs_dim: int, n_actions: int, device: str = "cuda"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_masks = np.ones((self.buffer_size, self.n_actions), dtype=bool)
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: Optional[np.ndarray] = None,
        **kwargs,  # 兼容旧调用传入clf_feat等参数
    ):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        if action_mask is not None:
            self.action_masks[self.ptr] = action_mask

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_gae_returns(self, last_value: float, gamma: float, lam: float):
        size = self.buffer_size if self.full else self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values[:size]

    def get_tensors(self, device: str = "cpu"):
        size = self.buffer_size if self.full else self.ptr
        return {
            "obs": torch.FloatTensor(self.observations[:size]).to(device),
            "actions": torch.LongTensor(self.actions[:size]).to(device),
            "log_probs": torch.FloatTensor(self.log_probs[:size]).to(device),
            "advantages": torch.FloatTensor(self.advantages).to(device),
            "returns": torch.FloatTensor(self.returns).to(device),
            "action_masks": torch.BoolTensor(self.action_masks[:size]).to(device),
        }

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.ptr
