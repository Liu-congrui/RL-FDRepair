"""
特征提取模块
===========
为RL策略网络提供结构化特征向量。

特征设计：
  GROUP_FEATURES(8) + RHS_FEATURES(2) + LHS_TOP3(3×4) = 22维

  GROUP: 冲突组统计信息
  RHS: 最高频候选的频率比 + 竞争度（次高频/最高频）
  LHS: top3候选各4维（证据匹配度、频率、影响分、归一化频率）

排序规则：
  LHS候选按"证据匹配优先、频率次之"排序，取top3。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from collections import Counter

from .fd_utils import ConflictGroup, FD, get_evidence_cols, rank_lhs_alts_by_evidence


# ============================================================
# 特征维度常量
# 修改这里需要同步修改policy.py中的input_dim
# ============================================================

GROUP_FEATURES = 8       # 组级别全局特征
EVIDENCE_FEATURES = 4    # 全局证据特征：[rhs_evidence_alignment, best_candidate_evidence, evidence_gap, evidence_coverage]
RHS_FEATURES = 4         # RHS整体特征：[最高频占比, 竞争度, 多数值LHS一致性, 少数值LHS一致性]
LHS_TOP_K = 3            # LHS候选保留top3
LHS_CANDIDATE_FEATURES = 5  # 每个LHS候选：[evidence_score, norm_freq, conflict_reduce, global_consistency, similarity_to_dirty]
ROW_LOCK_FEATURES = 4
TARGET_ROW_FEATURES = 2  # 全 FD 冲突总数 + 条件证据一致性

FEATURE_DIM = GROUP_FEATURES + EVIDENCE_FEATURES + RHS_FEATURES + LHS_TOP_K * LHS_CANDIDATE_FEATURES + ROW_LOCK_FEATURES + TARGET_ROW_FEATURES  # 37


class FeatureExtractor:
    """
    特征提取器。

    使用方式：
      fe = FeatureExtractor(global_df, fds)
      state_vec = fe.extract(conflict_group, current_df, step, max_steps)
    """

    def __init__(self, global_df: pd.DataFrame, fds: List[FD],
                 raw_df: Optional[pd.DataFrame] = None):
        self.global_df = global_df
        self._raw_df = raw_df if raw_df is not None else global_df
        self.fds = fds
        self._precompute_global_stats()

    def _precompute_global_stats(self):
        """预计算全局统计"""
        self._global_value_counts: Dict[str, Counter] = {}
        for col in self.global_df.columns:
            self._global_value_counts[col] = Counter(self.global_df[col].tolist())

        self._global_np: Dict[str, np.ndarray] = {
            col: self.global_df[col].to_numpy() for col in self.global_df.columns
        }

        # 证据缓存：(lhs_col, lhs_val, evidence_col) → (Counter, total)
        self._evidence_cache: Dict = {}
        ref = self._raw_df
        for fd in self.fds:
            col_map = {c.lower(): c for c in ref.columns}
            lhs_col = fd.lhs_cols[0] if fd.lhs_cols[0] in ref.columns \
                else col_map.get(fd.lhs_cols[0].lower())
            if not lhs_col:
                continue
            rhs_col = fd.rhs_col if fd.rhs_col in ref.columns \
                else col_map.get(fd.rhs_col.lower())
            fd_cols = {c.lower() for c in fd.lhs_cols} | {fd.rhs_col.lower()}
            evidence_cols = [c for c in ref.columns if c.lower() not in fd_cols][:5]
            for lhs_val, grp in ref.groupby(lhs_col, sort=False):
                if rhs_col and rhs_col in grp.columns and len(grp) > 0:
                    rhs_counts = Counter(grp[rhs_col].tolist())
                    majority_rhs_cnt = max(rhs_counts.values())
                    purity = majority_rhs_cnt / len(grp)
                else:
                    purity = 1.0
                for ecol in evidence_cols:
                    key = (lhs_col, lhs_val, ecol)
                    raw_counter = Counter(grp[ecol].tolist())
                    if purity >= 0.9:
                        c = raw_counter
                    else:
                        c = Counter({v: cnt * purity for v, cnt in raw_counter.items()})
                    self._evidence_cache[key] = (c, sum(c.values()))

        self._col_map_glo: Dict[str, str] = {c.lower(): c for c in self.global_df.columns}

    def extract(
        self,
        cg: ConflictGroup,
        current_df: pd.DataFrame,
        step: int,
        max_steps: int,
        lock_step_count: int = 0,
        affected_fd_count: int = 0,
        row_lhs_degree: int = 0,
        cascade_depth: int = 0,
        total_conflict_count: float = 0.0,
        conditional_evidence_consistency: float = 0.5,
        lhs_alts_list: Optional[List] = None,  # 预计算缓存
        lite: bool = False,  # True = 跳过证据列和 LHS 排名（行选择阶段）
    ) -> np.ndarray:
        """
        提取状态特征向量，shape=(FEATURE_DIM,)。
        """
        features = []

        # ── 1. 组级别全局特征 (8维) ─────────────────────────────
        group_size = len(cg.row_indices)
        n_rhs_distinct = len(cg.rhs_candidates)
        majority_count = max(cg.rhs_candidates.values()) if cg.rhs_candidates else 0
        minority_count = min(cg.rhs_candidates.values()) if cg.rhs_candidates else 0
        total_rhs = sum(cg.rhs_candidates.values())
        conflict_ratio = (group_size - majority_count) / max(group_size, 1)
        step_ratio = step / max(max_steps, 1)
        n_lhs_alts = len(cg.lhs_alternatives)

        group_feat = [
            float(group_size) / 20.0,
            float(n_rhs_distinct) / 8.0,
            float(majority_count) / max(group_size, 1),
            float(minority_count) / max(group_size, 1),
            conflict_ratio,
            step_ratio,
            float(n_lhs_alts) / 20.0,
            float(cg.has_conflict),
        ]
        assert len(group_feat) == GROUP_FEATURES
        features.extend(group_feat)

        # ── 2. RHS全局特征 (4维) ────────────────────────────────
        # 最高频占比
        max_freq_ratio = float(majority_count) / max(total_rhs, 1)
        # 竞争度：次高频/最高频（越接近1说明两个候选势均力敌，越不确定）
        sorted_rhs = sorted(cg.rhs_candidates.values(), reverse=True)
        second_count = sorted_rhs[1] if len(sorted_rhs) > 1 else 0
        competition = float(second_count) / max(majority_count, 1)

        # 多数值/少数值的 LHS 一致性：该 RHS 值出现时有多少比例是与当前 LHS 值一起出现的
        # 正确值（与当前 LHS 一致）→ 一致性高；错误值（来自其他 LHS 组）→ 一致性低
        col_map_cur_rhs = {c.lower(): c for c in current_df.columns}
        rhs_col_for_freq = cg.fd.rhs_col if cg.fd.rhs_col in current_df.columns \
            else col_map_cur_rhs.get(cg.fd.rhs_col.lower())
        lhs_cols_for_cons = [c if c in current_df.columns else col_map_cur_rhs.get(c.lower(), c)
                             for c in cg.fd.lhs_cols]
        if rhs_col_for_freq and rhs_col_for_freq in current_df.columns and cg.rhs_candidates:
            sorted_cands = sorted(cg.rhs_candidates.keys(),
                                  key=lambda v: cg.rhs_candidates[v], reverse=True)
            majority_rhs_val = sorted_cands[0]
            minority_rhs_val = sorted_cands[1] if len(sorted_cands) > 1 else sorted_cands[0]

            def _lhs_consistency(rhs_val: Any) -> float:
                # 全局出现次数
                global_mask = current_df[rhs_col_for_freq] == rhs_val
                global_count = global_mask.sum()
                if global_count == 0:
                    return 0.0
                # 与当前 LHS 值共同出现的次数
                lhs_mask = global_mask.copy()
                for col, val in zip(lhs_cols_for_cons, cg.lhs_value):
                    if col in current_df.columns:
                        lhs_mask = lhs_mask & (current_df[col] == val)
                co_count = lhs_mask.sum()
                return float(co_count) / float(global_count)

            majority_lhs_consistency = _lhs_consistency(majority_rhs_val)
            minority_lhs_consistency = _lhs_consistency(minority_rhs_val)
        else:
            majority_lhs_consistency = 0.0
            minority_lhs_consistency = 0.0

        rhs_feat = [max_freq_ratio, competition, majority_lhs_consistency, minority_lhs_consistency]
        assert len(rhs_feat) == RHS_FEATURES
        features.extend(rhs_feat)

        # ── 2.5 全局 EVIDENCE 特征 (4维) ────────────────────────────────
        # lite 模式：跳过昂贵的证据列计算和 LHS 排名，填入中性值
        if lite:
            features.extend([0.5, 0.5, 0.0, 0.5])  # EVIDENCE
            features.extend([0.0] * (LHS_TOP_K * LHS_CANDIDATE_FEATURES))  # LHS top3 padding
        else:
            # 确定eject_row（少数派行）
            col_map_cur = {c.lower(): c for c in current_df.columns}
            rhs_col_name = cg.fd.rhs_col if cg.fd.rhs_col in current_df.columns \
                else col_map_cur.get(cg.fd.rhs_col.lower())

            rows = cg.row_indices if cg.row_indices else [0]
            if rhs_col_name and cg.rhs_candidates:
                majority_rhs = max(cg.rhs_candidates, key=cg.rhs_candidates.get)
                minority_rows = [r for r in rows
                                 if current_df.at[r, rhs_col_name] != majority_rhs]
                eject_row = minority_rows[0] if minority_rows else rows[0]
            else:
                eject_row = rows[0]

            # 计算 RHS 多数值与证据列的对齐度
            rhs_evidence_alignment, evidence_count = self._compute_rhs_evidence_alignment(cg, current_df, eject_row)

            # 获取 LHS 候选列表（按证据匹配度排序）—— 支持外部缓存
            if lhs_alts_list is None:
                lhs_alts_list = rank_lhs_alts_by_evidence(
                    current_df, cg.fd, self.fds, cg.lhs_alternatives, eject_row
                )

            # 获取 top1 候选的证据匹配度
            best_candidate_evidence = 0.0
            if lhs_alts_list:
                _, _, best_candidate_evidence = lhs_alts_list[0]

            # 计算 evidence_gap：
            # 正值表示 RHS 多数值与证据列匹配度更高 → 倾向 RHS_UNIFY
            # 负值表示候选 LHS 与证据列匹配度更高 → 倾向 LHS_EJECT
            evidence_gap = rhs_evidence_alignment - best_candidate_evidence

            # 计算 evidence_coverage：参与计算的有效证据列数量 / 总证据列数量
            evidence_cols = get_evidence_cols(cg.fd, self.fds, list(current_df.columns), df=current_df)
            evidence_coverage = float(evidence_count) / max(len(evidence_cols), 1) if evidence_cols else 0.0

            evidence_feat = [
                rhs_evidence_alignment,
                best_candidate_evidence,
                evidence_gap,
                evidence_coverage,
            ]
            assert len(evidence_feat) == EVIDENCE_FEATURES
            features.extend(evidence_feat)

            # ── 3. LHS top3候选特征 (3×5=15维) ─────────────────────
            total_lhs_freq = sum(cg.lhs_alternatives.values()) if cg.lhs_alternatives else 1

            lhs_feat = []
            for i in range(LHS_TOP_K):
                if i < len(lhs_alts_list):
                    lhs_val, cnt, evidence_score = lhs_alts_list[i]
                    norm_freq = float(cnt) / max(total_lhs_freq, 1)

                    # 新增：计算 conflict_reduce 和 global_consistency
                    conflict_reduce = self._compute_conflict_reduce_score(lhs_val, eject_row, current_df, cg.fd)
                    global_consistency = self._compute_global_consistency_score(lhs_val, eject_row, current_df, self.fds)

                    # 新增：计算候选值与脏值的相似度
                    similarity_to_dirty = self._compute_similarity_to_dirty(lhs_val, eject_row, current_df, cg.fd)

                    lhs_feat.extend([evidence_score, norm_freq, conflict_reduce, global_consistency, similarity_to_dirty])
                else:
                    lhs_feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # padding

            assert len(lhs_feat) == LHS_TOP_K * LHS_CANDIDATE_FEATURES
            features.extend(lhs_feat)

        # ── 4. 行锁级联特征 (4维) ─────────────────────────────
        row_lock_feat = [
            float(lock_step_count) / max(max_steps, 1),
            float(affected_fd_count) / 10.0,
            float(row_lhs_degree) / 10.0,
            float(cascade_depth) / 5.0,
        ]
        features.extend(row_lock_feat)

        # ── 5. 目标行特征 (2维) ────────────────────────────
        target_row_feat = [
            float(total_conflict_count) / 50.0,  # 归一化
            float(conditional_evidence_consistency),
        ]
        features.extend(target_row_feat)

        vec = np.array(features, dtype=np.float32)
        assert vec.shape[0] == FEATURE_DIM, f"Feature dim mismatch: {vec.shape[0]} != {FEATURE_DIM}"
        return vec

    def _lhs_eject_impact_score(
        self, target_lhs: tuple, cg: ConflictGroup, current_df: pd.DataFrame
    ) -> float:
        """目标LHS组的干净程度（越干净=1.0，有冲突=低分）"""
        fd = cg.fd
        col_map = {c.lower(): c for c in current_df.columns}
        lhs_cols = []
        for col in fd.lhs_cols:
            if col in current_df.columns:
                lhs_cols.append(col)
            elif col.lower() in col_map:
                lhs_cols.append(col_map[col.lower()])
            else:
                return 0.5

        bool_mask = np.ones(len(current_df), dtype=bool)
        for col, val in zip(lhs_cols, target_lhs):
            bool_mask &= (current_df[col].to_numpy() == val)

        target_rows = current_df.iloc[bool_mask]
        rhs_col = fd.rhs_col if fd.rhs_col in current_df.columns else col_map.get(fd.rhs_col.lower())
        if not rhs_col:
            return 0.5

        rhs_distinct = target_rows[rhs_col].nunique()
        return 1.0 / max(rhs_distinct, 1)

    def _evidence_match_score(
        self, lhs_value: tuple, cg: ConflictGroup, current_df: pd.DataFrame, row_idx: int
    ) -> float:
        """
        候选LHS相对于当前LHS的证据列匹配分（log-ratio映射到0~1）。
        >0.5 表示候选LHS更合理；<0.5 表示候选LHS更差。
        """
        fd = cg.fd
        col_map_cur = {c.lower(): c for c in current_df.columns}
        col_map_glo = self._col_map_glo
        lhs_col_name = fd.lhs_cols[0] if fd.lhs_cols[0] in self.global_df.columns \
            else col_map_glo.get(fd.lhs_cols[0].lower())
        if not lhs_col_name:
            return 0.5

        fd_cols = {c.lower() for c in fd.lhs_cols} | {fd.rhs_col.lower()}
        evidence_cols = [c for c in current_df.columns if c.lower() not in fd_cols][:5]
        if not evidence_cols:
            return 0.5

        log_ratios = []
        for ecol in evidence_cols:
            cur_col = ecol if ecol in current_df.columns else col_map_cur.get(ecol.lower())
            glo_col = ecol if ecol in self.global_df.columns else col_map_glo.get(ecol.lower())
            if not cur_col or not glo_col:
                continue
            try:
                cur_val = current_df.at[row_idx, cur_col]
                key_cand = (lhs_col_name, lhs_value[0], glo_col)
                key_cur = (lhs_col_name, cg.lhs_value[0], glo_col)
                entry_cand = self._evidence_cache.get(key_cand)
                entry_cur = self._evidence_cache.get(key_cur)
                if entry_cand is None or entry_cur is None:
                    continue
                counter_cand, total_cand = entry_cand
                counter_cur, total_cur = entry_cur
                p_cand = counter_cand.get(cur_val, 0) / max(total_cand, 1)
                p_cur = counter_cur.get(cur_val, 0) / max(total_cur, 1)
                log_ratios.append(np.log(max(p_cand, 1e-6)) - np.log(max(p_cur, 1e-6)))
            except Exception:
                continue

        if not log_ratios:
            return 0.5
        avg_log_ratio = float(np.mean(log_ratios))
        return float(1.0 / (1.0 + np.exp(-avg_log_ratio * 3)))

    def _compute_rhs_evidence_alignment(
        self, cg: ConflictGroup, current_df: pd.DataFrame, eject_row: int
    ) -> Tuple[float, int]:
        """
        计算 RHS 多数值与证据列的对齐度。

        返回：(alignment_score, evidence_count)
          - alignment_score: [0, 1]，RHS 多数值与证据列的匹配度
          - evidence_count: 参与计算的有效证据列数量

        逻辑：
        - 如果 RHS 多数值与证据列匹配度高 → RHS 可能是对的 → 倾向 RHS_UNIFY
        - 如果 RHS 多数值与证据列匹配度低 → RHS 可能是错的 → 倾向 LHS_EJECT
        """
        fd = cg.fd
        col_map_cur = {c.lower(): c for c in current_df.columns}
        col_map_glo = self._col_map_glo

        rhs_col_name = fd.rhs_col if fd.rhs_col in current_df.columns \
            else col_map_cur.get(fd.rhs_col.lower())
        if not rhs_col_name:
            return 0.5, 0

        # 获取 RHS 多数值
        if not cg.rhs_candidates:
            return 0.5, 0
        majority_rhs = max(cg.rhs_candidates, key=cg.rhs_candidates.get)

        # 获取证据列（FD图 + Cramér's V 独立性检验）
        evidence_cols = get_evidence_cols(fd, self.fds, list(current_df.columns), df=current_df)
        if not evidence_cols:
            return 0.5, 0

        # eject_row 在证据列上的值（跳过 nan）
        eject_evidence = {}
        for ecol in evidence_cols:
            if ecol in current_df.columns:
                val = current_df.at[eject_row, ecol]
                if not (isinstance(val, float) and np.isnan(val)):
                    eject_evidence[ecol] = val

        if not eject_evidence:
            return 0.5, 0

        # 对于 RHS 多数值对应的行，计算其证据列与 eject_row 的匹配度
        mask = np.ones(len(current_df), dtype=bool)
        for col, val in zip([c if c in current_df.columns else col_map_cur.get(c.lower(), c) for c in fd.lhs_cols],
                           cg.lhs_value):
            if col in current_df.columns:
                mask &= (current_df[col].to_numpy() == val)

        # 在该 LHS 组中，找 RHS 多数值的行
        majority_mask = mask & (current_df[rhs_col_name].to_numpy() == majority_rhs)
        if not majority_mask.any():
            return 0.5, 0

        # 计算这些行的证据列众数与 eject_row 的匹配比例
        match_count = 0
        valid_evidence = 0
        for ecol, eject_val in eject_evidence.items():
            if ecol in current_df.columns:
                group_vals = current_df.loc[majority_mask, ecol].to_numpy()
                if len(group_vals) > 0:
                    mode_val = Counter(group_vals).most_common(1)[0][0]
                    if isinstance(mode_val, float) and np.isnan(mode_val):
                        continue
                    valid_evidence += 1
                    if mode_val == eject_val:
                        match_count += 1

        alignment_score = float(match_count) / max(valid_evidence, 1)
        return alignment_score, valid_evidence

    def _compute_conflict_reduce_score(
        self, target_lhs: tuple, eject_row: int, current_df: pd.DataFrame, fd: FD
    ) -> float:
        """
        计算选择 target_lhs 后能消除多少 FD 冲突（简化版）。

        返回：[0, 1]，表示冲突消除比例

        简化策略：
        - 如果 target_lhs 对应的行组没有冲突，返回 1.0
        - 否则返回 0.0
        """
        col_map = {c.lower(): c for c in current_df.columns}
        lhs_cols = [c if c in current_df.columns else col_map.get(c.lower(), c) for c in fd.lhs_cols]
        rhs_col = fd.rhs_col if fd.rhs_col in current_df.columns else col_map.get(fd.rhs_col.lower())

        if not rhs_col:
            return 0.0

        # 检查 target_lhs 对应的行组是否有冲突
        mask = np.ones(len(current_df), dtype=bool)
        for col, val in zip(lhs_cols, target_lhs):
            if col in current_df.columns:
                mask &= (current_df[col].to_numpy() == val)

        if not mask.any():
            return 0.0

        target_rhs_values = current_df.loc[mask, rhs_col].nunique()
        return 0.0 if target_rhs_values > 1 else 1.0

    def _compute_global_consistency_score(
        self, target_lhs: tuple, eject_row: int, current_df: pd.DataFrame, all_fds: List[FD]
    ) -> float:
        """
        计算修复后与其他 FD 的一致性（超简化版）。

        返回：[0, 1]，表示一致性比例

        简化策略：
        - 直接返回 0.5（中立值）
        - 让模型通过其他特征学习
        """
        return 0.5

    def _compute_similarity_to_dirty(
        self, candidate_lhs: tuple, eject_row: int, current_df: pd.DataFrame, fd: FD
    ) -> float:
        """
        计算候选 LHS 值与脏值的字符串相似度。

        返回：[0, 1]，表示相似度

        逻辑：
        - 获取当前行的脏 LHS 值
        - 计算候选值与脏值的相似度（使用 SequenceMatcher）
        - 返回归一化的相似度分数

        直觉：
        - 如果候选值与脏值相似，说明脏值可能是候选值的小错误（如打字错误）
        - 这种情况下候选值很可能是正确的
        """
        try:
            col_map = {c.lower(): c for c in current_df.columns}
            lhs_cols = [c if c in current_df.columns else col_map.get(c.lower(), c) for c in fd.lhs_cols]

            # 获取当前行的脏 LHS 值
            dirty_lhs_values = []
            for col in lhs_cols:
                if col in current_df.columns:
                    dirty_lhs_values.append(str(current_df.at[eject_row, col]))

            if not dirty_lhs_values:
                return 0.5

            # 计算候选值与脏值的相似度
            dirty_lhs_str = tuple(dirty_lhs_values)
            candidate_str = tuple(str(v) for v in candidate_lhs)

            # 逐个比较每个列的相似度，取平均
            similarities = []
            for dirty_val, cand_val in zip(dirty_lhs_str, candidate_str):
                matcher = SequenceMatcher(None, dirty_val, cand_val)
                ratio = matcher.ratio()  # 返回 [0, 1]
                similarities.append(ratio)

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.5
            return float(avg_similarity)
        except Exception:
            return 0.5

    def _compute_evidence_col_match_score(
        self, evidence_col: str, cg: ConflictGroup, current_df: pd.DataFrame, eject_row: int
    ) -> float:
        """
        计算单个证据列对于RHS多数值的匹配度。

        返回：[0, 1]，表示匹配度

        逻辑：
        - 获取RHS多数值对应的行组
        - 计算该证据列在这个行组中的主要值
        - 计算当前行的该证据列值与主要值的匹配度
        - 返回匹配度分数

        这样模型可以学习到每个证据列的重要性：
        - 重要的证据列（如city）会有高的匹配度
        - 不重要的证据列（如state）会有低的匹配度
        """
        try:
            fd = cg.fd
            col_map = {c.lower(): c for c in current_df.columns}

            # 获取RHS列名
            rhs_col = fd.rhs_col if fd.rhs_col in current_df.columns \
                else col_map.get(fd.rhs_col.lower())
            if not rhs_col:
                return 0.5

            # 获取RHS多数值
            majority_rhs = max(cg.rhs_candidates, key=cg.rhs_candidates.get)

            # 获取RHS多数值对应的行
            majority_rows = current_df[current_df[rhs_col] == majority_rhs]
            if len(majority_rows) == 0:
                return 0.5

            # 获取该证据列在多数值行组中的主要值
            evidence_col_actual = evidence_col if evidence_col in current_df.columns \
                else col_map.get(evidence_col.lower())
            if not evidence_col_actual:
                return 0.5

            # 计算主要值
            value_counts = majority_rows[evidence_col_actual].value_counts()
            if len(value_counts) == 0:
                return 0.5

            main_value = value_counts.index[0]
            main_count = value_counts.values[0]
            main_ratio = main_count / len(majority_rows)

            # 获取当前行的该证据列值
            current_value = current_df.at[eject_row, evidence_col_actual]

            # 计算匹配度
            if current_value == main_value:
                # 完全匹配：返回主要值的比例
                return float(main_ratio)
            else:
                # 不匹配：返回0
                return 0.0
        except Exception:
            return 0.5




