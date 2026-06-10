"""
FD工具模块
=========
职责：
  1. 解析函数依赖（FD）定义
  2. 根据FD从DataFrame中构建冲突组
  3. 统计每个冲突组的左部/右部候选值
  4. 执行动作后更新DataFrame状态

假设：
  - 正确值一定存在于当前冲突组的候选值集合中
  - 正确值不一定是多数值（这是RL需要学习的核心）
  - 一个FD写成 (lhs_cols, rhs_col)，当前只支持单右部列
    （未来扩展：支持复合右部）

扩展点：
  - 多FD同时处理：迭代每条FD构建冲突组列表
  - 复合右部：将rhs_col改为rhs_cols列表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
from collections import Counter


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class FD:
    """函数依赖：lhs_cols -> rhs_col"""
    lhs_cols: List[str]   # 左部列名列表
    rhs_col: str          # 右部列名（当前只支持单列）

    def __repr__(self):
        lhs = ", ".join(self.lhs_cols)
        return f"FD({lhs} -> {self.rhs_col})"

    def __hash__(self):
        return hash((tuple(self.lhs_cols), self.rhs_col))

    def __eq__(self, other):
        if not isinstance(other, FD):
            return False
        return self.lhs_cols == other.lhs_cols and self.rhs_col == other.rhs_col


@dataclass
class ConflictGroup:
    """
    一个冲突组：具有相同左部值、右部出现多个不同值的行集合

    属性：
      fd           : 对应的函数依赖
      lhs_value    : 左部的具体取值（tuple形式，对应lhs_cols的顺序）
      row_indices  : 该组包含的行索引（原始DataFrame中的index）
      rhs_candidates: 右部候选值 -> 出现次数
      lhs_candidates: 左部候选值（当前为单一值，扩展用）
      current_df   : 对当前状态的引用快照（执行动作时会更新）

    简化假设：
      - lhs_candidates目前固定为当前组的lhs_value
      - 未来扩展：考虑左部修复时从全局挖掘候选
    """
    fd: FD
    lhs_value: tuple          # e.g. ("ZhangSan",) 或 ("ZhangSan", "BJ")
    row_indices: List[int]    # 行索引
    rhs_candidates: Dict[Any, int] = field(default_factory=dict)  # value -> count
    lhs_alternatives: Dict[tuple, int] = field(default_factory=dict)  # 左部其他候选（来自全局）

    @property
    def has_conflict(self) -> bool:
        return len(self.rhs_candidates) > 1

    @property
    def majority_rhs(self) -> Any:
        """右部多数值"""
        if not self.rhs_candidates:
            return None
        return max(self.rhs_candidates, key=self.rhs_candidates.get)

    @property
    def minority_rhs_values(self) -> List[Any]:
        """右部少数值列表"""
        if not self.rhs_candidates:
            return []
        max_count = self.rhs_candidates[self.majority_rhs]
        return [v for v, c in self.rhs_candidates.items() if c < max_count]

    def rhs_candidate_list(self) -> List[Any]:
        """按频率降序排列的右部候选值列表"""
        return sorted(self.rhs_candidates.keys(),
                      key=lambda v: self.rhs_candidates[v], reverse=True)

    def __repr__(self):
        return (f"ConflictGroup(fd={self.fd}, lhs={self.lhs_value}, "
                f"rows={self.row_indices}, rhs_cands={self.rhs_candidates})")


# ============================================================
# FD解析
# ============================================================

def parse_fd(fd_str: str) -> FD:
    """
    解析FD字符串。
    格式示例：
      "A -> B"
      "A, C -> B"
    """
    parts = fd_str.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid FD format: {fd_str!r}. Expected 'LHS -> RHS'.")
    lhs_str, rhs_str = parts
    lhs_cols = [c.strip() for c in lhs_str.split(",")]
    rhs_col = rhs_str.strip()
    return FD(lhs_cols=lhs_cols, rhs_col=rhs_col)


def parse_fds(fd_list: List[str]) -> List[FD]:
    """批量解析FD列表"""
    return [parse_fd(s) for s in fd_list]


# ============================================================
# 冲突组构建
# ============================================================

def build_conflict_groups(df: pd.DataFrame, fd: FD,
                          global_df: Optional[pd.DataFrame] = None) -> List[ConflictGroup]:
    """
    根据给定FD从df中构建所有冲突组。
    """
    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}

    lhs_cols = []
    for col in fd.lhs_cols:
        if col in df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            return []  # 列不存在，返回空列表

    rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
    if not rhs_col or rhs_col not in df.columns:
        return []

    ref_df = global_df if global_df is not None else df
    conflict_groups = []

    # 按左部值分组
    grouped = df.groupby(lhs_cols, sort=False)
    for lhs_val, group in grouped:
        # pandas groupby单列时lhs_val是标量，多列时是tuple
        if isinstance(lhs_val, (list, tuple)):
            lhs_key = tuple(lhs_val)
        else:
            lhs_key = (lhs_val,)

        rhs_vals = group[rhs_col].tolist()
        rhs_counter = Counter(rhs_vals)

        # 只保留有冲突的组
        if len(rhs_counter) <= 1:
            continue

        row_idx = list(group.index)

        # 从全局df挖掘左部替代候选（用于左部修复）
        # 简化：取全局df中所有出现的lhs组合及其频率
        lhs_alts = _compute_lhs_alternatives(ref_df, fd, lhs_key)

        cg = ConflictGroup(
            fd=fd,
            lhs_value=lhs_key,
            row_indices=row_idx,
            rhs_candidates=dict(rhs_counter),
            lhs_alternatives=lhs_alts,
        )
        conflict_groups.append(cg)

    return conflict_groups


def build_all_conflict_groups(df: pd.DataFrame,
                              fds: List[FD]) -> List[Tuple[FD, List[ConflictGroup]]]:
    """
    对所有FD构建冲突组。

    返回：
      [(fd, [conflict_group, ...]), ...]
    """
    result = []
    for fd in fds:
        try:
            cgs = build_conflict_groups(df, fd, global_df=df)
            result.append((fd, cgs))
        except KeyError as e:
            print(f"[WARNING] Skipping FD {fd}: {e}")
    return result


# Cramér's V 缓存：避免重复计算相同列对的相关系数
_cramers_v_cache: Dict[tuple, float] = {}

# 证据列缓存：每个 FD 的证据列集合（列名不随数据变化，预计算一次）
_evidence_cols_cache: Dict[tuple, List[str]] = {}


def clear_evidence_cols_cache():
    """清空证据列缓存"""
    _evidence_cols_cache.clear()


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    计算两个分类变量的Cramér's V相关系数 (0-1)。

    返回：
      相关系数 [0, 1]，0表示完全独立，1表示完全相关
    """
    try:
        # 用列名 + 值哈希做缓存 key
        cache_key = (x.name, y.name, hash(tuple(x.to_numpy())), hash(tuple(y.to_numpy())))
        if cache_key in _cramers_v_cache:
            return _cramers_v_cache[cache_key]
    except Exception:
        cache_key = None

    try:
        from scipy.stats import chi2_contingency
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0 or n == 0:
            result = 0.0
        else:
            result = float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        result = 0.5  # 计算失败时返回中立值

    if cache_key is not None:
        _cramers_v_cache[cache_key] = result
    return result


def clear_cramers_v_cache():
    """清空 Cramér's V 缓存（epoch 切换时调用）"""
    _cramers_v_cache.clear()


def select_evidence_cols_by_independence(
    df: pd.DataFrame,
    fd: FD,
    independence_threshold: float = 0.3,
    max_cols: int = 5
) -> List[str]:
    """
    基于 Cramér's V 相关系数选择独立于 LHS 和 RHS 的证据列。

    原理：
      好的证据列应与当前FD的LHS和RHS都独立（不相关）——
      这样才能提供"外部视角"来判定哪一方是正确的。
      1. 计算每个非FD列与LHS的 Cramér's V
      2. 计算每个非FD列与RHS的 Cramér's V
      3. 选择同时独立于LHS和RHS的列（相关系数 < threshold）
      4. 按独立性强度排序，取top-max_cols
    """
    fd_cols_lower = {c.lower() for c in fd.lhs_cols} | {fd.rhs_col.lower()}
    col_map = {c.lower(): c for c in df.columns}

    lhs_col = fd.lhs_cols[0]
    if lhs_col not in df.columns:
        lhs_col = col_map.get(lhs_col.lower(), lhs_col)

    rhs_col = fd.rhs_col
    if rhs_col not in df.columns:
        rhs_col = col_map.get(rhs_col.lower(), rhs_col)

    candidate_cols = [c for c in df.columns if c.lower() not in fd_cols_lower]

    if not candidate_cols or lhs_col not in df.columns or rhs_col not in df.columns:
        return candidate_cols[:max_cols]

    col_scores = []
    for col in candidate_cols:
        try:
            lhs_corr = cramers_v(df[lhs_col], df[col])
            rhs_corr = cramers_v(df[rhs_col], df[col])
            if lhs_corr < independence_threshold and rhs_corr < independence_threshold:
                independence_score = 1.0 - max(lhs_corr, rhs_corr)
                col_scores.append((col, independence_score))
        except Exception:
            continue

    col_scores.sort(key=lambda x: -x[1])

    # 如果没有满足阈值的列，降低阈值重试
    if not col_scores:
        for c in candidate_cols:
            try:
                lhs_corr = cramers_v(df[lhs_col], df[c])
                rhs_corr = cramers_v(df[rhs_col], df[c])
                col_scores.append((c, 1.0 - max(lhs_corr, rhs_corr)))
            except Exception:
                col_scores.append((c, 0.5))
        col_scores.sort(key=lambda x: -x[1])

    return [c for c, _ in col_scores[:max_cols]]


def filter_noise_columns(
    df: pd.DataFrame,
    fd: FD,
    noise_threshold: float = 0.15,
    max_cols: int = 5,
) -> List[str]:
    """
    从候选列中滤掉与LHS和RHS均独立的噪声列。

    原理（与 select_evidence_cols_by_independence 相反）：
      1. 取所有非FD列作为初始候选
      2. 计算每列与LHS、与RHS的Cramér's V
      3. 去掉与LHS和RHS都独立（V < noise_threshold）的列
      4. 剩余的列至少与LHS或RHS之一有关联，具有证据价值
      5. 按相关性强度（max(lhs_corr, rhs_corr)）降序排列，取top max_cols

    返回：按证据强度降序的过滤后列名列表
    """
    fd_cols_lower = {c.lower() for c in fd.lhs_cols} | {fd.rhs_col.lower()}
    col_map = {c.lower(): c for c in df.columns}

    lhs_col = fd.lhs_cols[0]
    if lhs_col not in df.columns:
        lhs_col = col_map.get(lhs_col.lower(), lhs_col)

    rhs_col = fd.rhs_col
    if rhs_col not in df.columns:
        rhs_col = col_map.get(rhs_col.lower(), rhs_col)

    candidate_cols = [c for c in df.columns if c.lower() not in fd_cols_lower]

    if not candidate_cols or lhs_col not in df.columns or rhs_col not in df.columns:
        return candidate_cols[:max_cols]

    evidence = []
    for col in candidate_cols:
        try:
            lhs_corr = cramers_v(df[lhs_col], df[col])
            rhs_corr = cramers_v(df[rhs_col], df[col])
            # 去掉与两者均独立的噪声列
            if lhs_corr < noise_threshold and rhs_corr < noise_threshold:
                continue
            evidence.append((col, max(lhs_corr, rhs_corr)))
        except Exception:
            evidence.append((col, 0.5))  # 计算失败则给中立分数

    # 按相关性强度降序排列
    evidence.sort(key=lambda x: -x[1])
    return [c for c, _ in evidence[:max_cols]]


def get_evidence_cols(fd: FD, all_fds: List[FD],
                      df_columns: List[str],
                      df: Optional[pd.DataFrame] = None,
                      max_cols: int = 5) -> List[str]:
    """
    从 FD 图中找出当前 FD 的证据列。

    优先级：
      1. 噪声过滤(V<0.4) ∩ FD图合并：FD图列排前，过滤列补充
      2. 纯噪声过滤兜底（FD图无结果时）
      3. FD图兜底（噪声过滤无结果时）
      4. 简单方法（排除FD列）

    FD图规则：遍历所有 FD，找与当前 FD 的 LHS/RHS 有直接关系的列：
      - 当前 RHS 出现在 other_fd 的 LHS → other_fd 的 RHS 是证据列
      - 当前 LHS 出现在 other_fd 的 RHS → other_fd 的 LHS 是证据列
      - 共享 RHS → other_fd 的 LHS 是证据列
      - 共享 LHS → other_fd 的 RHS 是证据列
    排除当前 FD 自身的 LHS 和 RHS 列。
    最终返回最多 max_cols 列。
    """
    # 检查缓存
    cache_key = (tuple(sorted(fd.lhs_cols)), fd.rhs_col, tuple(sorted(df_columns)), max_cols)
    if cache_key in _evidence_cols_cache:
        return _evidence_cols_cache[cache_key]

    fd_cols_lower = {c.lower() for c in fd.lhs_cols} | {fd.rhs_col.lower()}
    related_lower = set()
    for other_fd in all_fds:
        o_lhs = {c.lower() for c in other_fd.lhs_cols}
        o_rhs = {other_fd.rhs_col.lower()}
        if fd.rhs_col.lower() in o_lhs:
            related_lower |= o_rhs
        for lc in fd.lhs_cols:
            if lc.lower() in o_rhs:
                related_lower |= o_lhs
        if fd.rhs_col.lower() == other_fd.rhs_col.lower() and other_fd is not fd:
            related_lower |= o_lhs
        if o_lhs & {c.lower() for c in fd.lhs_cols} and other_fd is not fd:
            related_lower |= o_rhs
    related_lower -= fd_cols_lower

    col_lower_to_actual = {c.lower(): c for c in df_columns}
    evidence = [col_lower_to_actual[c] for c in related_lower if c in col_lower_to_actual]

    # 步骤1: 噪声过滤 + FD图合并
    #   - 噪声过滤去掉与LHS和RHS均独立的列（V < 0.4），按强度排序
    #   - FD图列提供语义区分力，排在最前面
    if df is not None:
        try:
            filtered = filter_noise_columns(df, fd, noise_threshold=0.4, max_cols=max_cols)
            if evidence:
                # FD图列排前 + 过滤列补充（去重），总体不超过 max_cols
                evidence_set = set(evidence)
                for c in filtered:
                    if c not in evidence_set:
                        evidence.append(c)
                        evidence_set.add(c)
                        if len(evidence) >= max_cols:
                            break
                result = evidence[:max_cols]
                _evidence_cols_cache[cache_key] = result
                return result
            elif filtered:
                result = filtered[:max_cols]
                _evidence_cols_cache[cache_key] = result
                return result
        except Exception:
            pass

    # 步骤2: FD图兜底
    if evidence:
        result = evidence[:max_cols]
        _evidence_cols_cache[cache_key] = result
        return result

    # 步骤3: Fallback
    result = [c for c in df_columns if c.lower() not in fd_cols_lower][:max_cols]
    _evidence_cols_cache[cache_key] = result
    return result


def rank_lhs_alts_by_evidence(
    df: pd.DataFrame,
    fd: FD,
    all_fds: List[FD],
    lhs_alts: Dict[tuple, int],
    eject_row_idx: int,
    _cache: Optional[Dict] = None,
) -> List[Tuple[tuple, int]]:
    """
    用证据列对 LHS 候选集重排序（加权版）。

    对每个候选 LHS 值，计算其组内行在证据列上与 eject_row 的加权匹配度。
    每个证据列的权重 = max(Cramér's V(col, LHS), Cramér's V(col, RHS))，
    即该列与 FD 的相关性越强，权重越大，避免弱相关列稀释匹配信号。

    返回排序后的 [(lhs_value, freq, evidence_score), ...] 列表。
    """
    if not lhs_alts:
        return []

    evidence_cols = get_evidence_cols(fd, all_fds, list(df.columns), df=df)
    if not evidence_cols:
        return sorted([(v, f, 0.0) for v, f in lhs_alts.items()], key=lambda x: -x[1])

    col_map = {c.lower(): c for c in df.columns}
    lhs_cols = [c if c in df.columns else col_map.get(c.lower(), c) for c in fd.lhs_cols]
    rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())

    # 计算每个证据列的权重：max(V(col, LHS第一列), V(col, RHS))
    lhs_col_0 = lhs_cols[0]
    evidence_weights = {}
    for ecol in evidence_cols:
        if ecol in df.columns:
            try:
                lhs_corr = cramers_v(df[lhs_col_0], df[ecol])
                rhs_corr = cramers_v(df[rhs_col], df[ecol]) if rhs_col and rhs_col in df.columns else 0.0
                evidence_weights[ecol] = max(lhs_corr, rhs_corr)
            except Exception:
                evidence_weights[ecol] = 0.5

    # eject_row 在证据列上的值（跳过 nan，nan 无法提供有效匹配信号）
    eject_evidence = {}
    for ecol in evidence_cols:
        if ecol in df.columns:
            val = df.at[eject_row_idx, ecol]
            if not (isinstance(val, float) and np.isnan(val)):
                eject_evidence[ecol] = val

    if not eject_evidence:
        return sorted([(v, f, 0.0) for v, f in lhs_alts.items()], key=lambda x: -x[1])

    # 预计算：用 numpy 数组加速 mask 操作
    df_arrays = {col: df[col].to_numpy() if col in df.columns else None for col in lhs_cols}
    evidence_arrays = {col: df[col].to_numpy() if col in df.columns else None for col in evidence_cols}

    scored = []
    for lhs_val, freq in lhs_alts.items():
        # 用 numpy 数组快速计算 mask
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(lhs_cols, lhs_val):
            arr = df_arrays[col]
            if arr is not None:
                mask &= (arr == val)

        if not mask.any():
            scored.append((lhs_val, freq, 0.0))
            continue

        # 加权匹配：每列匹配成功加 weight，分母为总 weight
        match_weight = 0.0
        total_weight = 0.0
        for ecol, eject_val in eject_evidence.items():
            arr = evidence_arrays[ecol]
            if arr is not None:
                group_vals = arr[mask]
                if len(group_vals) > 0:
                    mode_val = Counter(group_vals).most_common(1)[0][0]
                    # 跳过 nan 众数（nan 无法提供有效信号）
                    if isinstance(mode_val, float) and np.isnan(mode_val):
                        continue
                    w = evidence_weights.get(ecol, 0.5)
                    total_weight += w
                    if mode_val == eject_val:
                        match_weight += w
        match_score = match_weight / max(total_weight, 0.001)
        scored.append((lhs_val, freq, match_score))

    # 按匹配度降序，匹配度相同按频率降序
    scored.sort(key=lambda x: (-x[2], -x[1]))
    return [(v, f, s) for v, f, s in scored]


def _compute_lhs_alternatives(df: pd.DataFrame, fd: FD, lhs_key: tuple) -> Dict[tuple, int]:
    """
    从全局df中找出左部的其他候选值及其频率。
    用于左部修复时选择目标值。

    改进版：不仅包括已存在的lhs组合，还包括所有可能的单列值组合。
    """
    alts = {}

    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    lhs_cols = []
    for col in fd.lhs_cols:
        if col in df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            return {}

    # 方法1：已存在的lhs组合（原有逻辑）
    grouped = df.groupby(lhs_cols, sort=False, dropna=False)
    for val, group in grouped:
        if isinstance(val, (list, tuple)):
            key = tuple(val)
        else:
            key = (val,)
        # 跳过 nan LHS 值（nan 不可作为修复目标）
        if any(isinstance(v, float) and np.isnan(v) for v in key):
            continue
        if key != lhs_key:
            alts[key] = len(group)

    # 方法2：扩展到所有可能的单列值（新增）
    # 对于每个LHS列，取该列的所有唯一值
    if len(lhs_cols) == 1:
        # 单列LHS：直接取所有唯一值（跳过 nan）
        col = lhs_cols[0]
        for val in df[col].unique():
            if isinstance(val, float) and np.isnan(val):
                continue
            key = (val,)
            if key != lhs_key and key not in alts:
                alts[key] = 1  # 频率设为1（表示可能的候选）

    return alts


# ============================================================
# 动作执行 & 状态更新
# ============================================================

def apply_rhs_unify(df: pd.DataFrame, cg: ConflictGroup,
                    target_value: Any) -> pd.DataFrame:
    """
    动作模板A：右部统一
    将当前冲突组内所有行的rhs_col统一修改为target_value。

    返回修改后的DataFrame副本（不修改原始df）。

    注意：
      - target_value必须在cg.rhs_candidates中
      - 这是"选定候选值，其他行改成它"的操作
    """
    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    rhs_col = cg.fd.rhs_col if cg.fd.rhs_col in df.columns else col_map.get(cg.fd.rhs_col.lower(), cg.fd.rhs_col)

    df_new = df.copy()
    for idx in cg.row_indices:
        df_new.at[idx, rhs_col] = target_value
    return df_new


def apply_lhs_eject(df: pd.DataFrame, cg: ConflictGroup,
                    eject_row_idx: int, new_lhs_value: tuple) -> pd.DataFrame:
    """
    动作模板B：左部异常tuple移出
    将指定行的左部值修改为new_lhs_value，使其离开当前冲突组。

    参数：
      eject_row_idx : 要移出的行索引（必须在cg.row_indices中）
      new_lhs_value : 新的左部值（tuple，对应fd.lhs_cols）

    返回修改后的DataFrame副本。

    简化版限制：
      - new_lhs_value必须在cg.lhs_alternatives中（全局已存在的lhs值）
      - 未来扩展：允许创建新的lhs值（当前行本身的正确值）
    """
    if eject_row_idx not in cg.row_indices:
        raise ValueError(f"Row {eject_row_idx} not in conflict group {cg.lhs_value}")

    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    lhs_cols = []
    for col in cg.fd.lhs_cols:
        if col in df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            raise ValueError(f"Column {col} not found in DataFrame")

    df_new = df.copy()
    for col, val in zip(lhs_cols, new_lhs_value):
        df_new.at[eject_row_idx, col] = val
    return df_new


def refresh_conflict_group(df: pd.DataFrame, cg: ConflictGroup) -> ConflictGroup:
    """
    执行动作后，根据最新df刷新冲突组状态。
    返回新的ConflictGroup对象。
    """
    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    lhs_cols = []
    for col in cg.fd.lhs_cols:
        if col in df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            return ConflictGroup(fd=cg.fd, lhs_value=cg.lhs_value, row_indices=[], rhs_candidates={}, lhs_alternatives=cg.lhs_alternatives)

    rhs_col = cg.fd.rhs_col if cg.fd.rhs_col in df.columns else col_map.get(cg.fd.rhs_col.lower())
    if not rhs_col:
        return ConflictGroup(fd=cg.fd, lhs_value=cg.lhs_value, row_indices=[], rhs_candidates={}, lhs_alternatives=cg.lhs_alternatives)

    # 重新查找属于当前lhs_value的行
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in zip(lhs_cols, cg.lhs_value):
        mask &= (df[col] == val)
    new_rows = list(df[mask].index)
    new_rhs_counter = Counter(df.loc[new_rows, rhs_col].tolist()) if new_rows else {}

    return ConflictGroup(
        fd=cg.fd,
        lhs_value=cg.lhs_value,
        row_indices=new_rows,
        rhs_candidates=dict(new_rhs_counter),
        lhs_alternatives=cg.lhs_alternatives,
    )


# ============================================================
# 统计工具
# ============================================================

def count_fd_violations(df: pd.DataFrame, fd: FD) -> int:
    """计算当前df中某条FD的冲突数（违反该FD的tuple对数量）"""
    violations = 0

    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    lhs_cols = []
    for col in fd.lhs_cols:
        if col in df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            return 0  # 列不存在

    rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
    if not rhs_col or rhs_col not in df.columns:
        return 0

    try:
        grouped = df.groupby(lhs_cols, sort=False)
        for _, group in grouped:
            distinct_rhs = group[rhs_col].nunique()
            if distinct_rhs > 1:
                rhs_counts = Counter(group[rhs_col].tolist())
                majority_count = max(rhs_counts.values())
                violations += len(group) - majority_count
    except KeyError:
        return 0

    return violations


def count_all_fd_violations(df: pd.DataFrame, fds: List[FD]) -> Dict[FD, int]:
    """对所有FD计算冲突数"""
    return {fd: count_fd_violations(df, fd) for fd in fds}


def build_pseudo_clean_df(dirty_df: pd.DataFrame, fds: List[FD],
                          target_ratio: float = 0.7) -> pd.DataFrame:
    """
    从 dirty_df 中提取无冲突行作为伪干净参考。

    策略（优先级递减）：
      1. 主方案：完全无冲突行（不参与任何 FD 的冲突组），需达到 target_ratio
      2. 备用方案：不足时补充共识行（在冲突组中但 RHS 总与多数一致）
      3. 兜底方案：仍不足时 bootstrap 重采样补足

    共识行准确率：注错模型下，RHS 被改错的行必是少数派 → 被排除。
    实测 hospital 0.30 错误率下共识行 100% 正确。
    """
    target_rows = max(int(len(dirty_df) * target_ratio), 1)

    conflict_indices: Set[int] = set()
    minority_indices: Set[int] = set()

    col_map = {c.lower(): c for c in dirty_df.columns}

    for fd in fds:
        try:
            cgs = build_conflict_groups(dirty_df, fd)
        except KeyError:
            continue
        for cg in cgs:
            if not cg.has_conflict:
                continue
            conflict_indices.update(cg.row_indices)

            # 找该 CG 的少数派行
            rhs_col = fd.rhs_col if fd.rhs_col in dirty_df.columns \
                else col_map.get(fd.rhs_col.lower())
            if not rhs_col:
                continue
            majority = cg.majority_rhs
            for idx in cg.row_indices:
                if idx in dirty_df.index:
                    try:
                        if dirty_df.at[idx, rhs_col] != majority:
                            minority_indices.add(idx)
                    except (KeyError, ValueError):
                        continue

    # 主方案：完全干净行
    fully_clean = [i for i in dirty_df.index if i not in conflict_indices]

    if len(fully_clean) >= target_rows:
        return dirty_df.loc[fully_clean].copy()

    # 备用方案：补充共识行
    consensus = list(conflict_indices - minority_indices)
    all_clean = sorted(set(fully_clean) | set(consensus))

    if not all_clean:
        return dirty_df.copy()

    pseudo_clean = dirty_df.loc[all_clean].copy()

    # 兜底：bootstrap 补足
    if len(pseudo_clean) < target_rows:
        n_extra = target_rows - len(pseudo_clean)
        extra = pseudo_clean.sample(n=n_extra, replace=True, random_state=42)
        pseudo_clean = pd.concat([pseudo_clean, extra], ignore_index=True)

    return pseudo_clean


def resolve_fd_conflicts(df: pd.DataFrame, fds: List[FD]) -> pd.DataFrame:
    """
    消除DataFrame中的FD冲突（通过统一RHS值）。

    对于每个FD (lhs → rhs)：
    - 按lhs分组
    - 对于每个lhs值，如果有多个rhs值，统一为频率最高的
    - 修改其他行的rhs值

    参数：
      df: 输入DataFrame
      fds: FD列表

    返回：
      无冲突的DataFrame（所有行保留，只修改RHS值）
    """
    result_df = df.copy()
    col_map = {c.lower(): c for c in result_df.columns}

    for fd in fds:
        # 处理列名大小写
        lhs_cols = []
        for col in fd.lhs_cols:
            if col in result_df.columns:
                lhs_cols.append(col)
            elif col.lower() in col_map:
                lhs_cols.append(col_map[col.lower()])

        rhs_col = fd.rhs_col if fd.rhs_col in result_df.columns \
            else col_map.get(fd.rhs_col.lower())

        if not lhs_cols or not rhs_col:
            continue

        # 按lhs分组
        grouped = result_df.groupby(lhs_cols, sort=False)

        # 对每个lhs值，检查是否有多个rhs值
        for lhs_val, group in grouped:
            rhs_values = group[rhs_col].value_counts()

            if len(rhs_values) > 1:
                # 有冲突，统一为频率最高的
                main_rhs = rhs_values.index[0]

                # 找出这个lhs值对应的所有行的索引
                mask = pd.Series([True] * len(result_df), index=result_df.index)
                if isinstance(lhs_val, tuple):
                    for col, val in zip(lhs_cols, lhs_val):
                        mask &= (result_df[col] == val)
                else:
                    mask &= (result_df[lhs_cols[0]] == lhs_val)

                # 修改这些行的rhs值为main_rhs
                result_df.loc[mask, rhs_col] = main_rhs

    return result_df


def identify_pseudo_conflicts(dirty_df: pd.DataFrame,
                              clean_df: pd.DataFrame,
                              fds: List[FD]) -> Set[int]:
    """
    识别伪冲突（pseudo-conflicts）：在干净数据中也存在的冲突。

    伪冲突是指：某个冲突组在脏数据中有多个RHS值，但在干净数据中这个LHS值只对应一个RHS值。
    这意味着这个冲突是由于数据注入错误造成的，而不是真实的数据质量问题。

    返回：伪冲突组的索引集合（用于跳过修复）
    """
    pseudo_conflict_indices = set()
    col_map_dirty = {c.lower(): c for c in dirty_df.columns}
    col_map_clean = {c.lower(): c for c in clean_df.columns}

    for fd in fds:
        # 处理列名大小写
        lhs_cols_dirty = []
        for col in fd.lhs_cols:
            if col in dirty_df.columns:
                lhs_cols_dirty.append(col)
            elif col.lower() in col_map_dirty:
                lhs_cols_dirty.append(col_map_dirty[col.lower()])

        lhs_cols_clean = []
        for col in fd.lhs_cols:
            if col in clean_df.columns:
                lhs_cols_clean.append(col)
            elif col.lower() in col_map_clean:
                lhs_cols_clean.append(col_map_clean[col.lower()])

        rhs_col_dirty = fd.rhs_col if fd.rhs_col in dirty_df.columns \
            else col_map_dirty.get(fd.rhs_col.lower())

        rhs_col_clean = fd.rhs_col if fd.rhs_col in clean_df.columns \
            else col_map_clean.get(fd.rhs_col.lower())

        if not lhs_cols_dirty or not rhs_col_dirty or not rhs_col_clean or not lhs_cols_clean:
            continue

        # 在脏数据中找冲突组
        all_cgs = build_conflict_groups(dirty_df, fd, global_df=dirty_df)

        for cg in all_cgs:
            if not cg.has_conflict:
                continue

            # 检查这个冲突组在干净数据中是否也有冲突
            # 方法：对于这个冲突组中的每一行，检查在干净数据中是否存在相同LHS值的行
            # 如果存在，检查这些行的RHS值是否都相同

            # 获取这个冲突组的LHS值
            lhs_value = cg.lhs_value

            # 在干净数据中找所有具有相同LHS值的行
            mask = pd.Series([True] * len(clean_df), index=clean_df.index)
            for col, val in zip(lhs_cols_clean, lhs_value):
                mask &= (clean_df[col] == val)

            # 获取这些行的RHS值
            rhs_vals_in_clean = set(clean_df.loc[mask, rhs_col_clean].unique())

            # 如果干净数据中只有一个RHS值，这是伪冲突
            if len(rhs_vals_in_clean) <= 1:
                # 标记这个冲突组中的所有行
                for row_idx in cg.row_indices:
                    pseudo_conflict_indices.add(row_idx)

    return pseudo_conflict_indices


# ============================================================
# FD图构建与邻域计算
# ============================================================

def build_fd_graph(fds: List[FD]) -> Dict[FD, List[FD]]:
    """
    构建无向FD依赖图（邻接表）。

    两个FD相连的条件（满足任一即可）：
      - 共享LHS列（交集非空）
      - 一个FD的RHS列在另一个FD的LHS中
      - 共享相同的RHS列

    使用大小写不敏感比较。

    参数：
      fds: FD列表

    返回：
      {fd: [neighbor_fd, ...]}
    """
    graph: Dict[FD, List[FD]] = {fd: [] for fd in fds}

    for i, fd1 in enumerate(fds):
        lhs1_lower = {c.lower() for c in fd1.lhs_cols}
        rhs1_lower = fd1.rhs_col.lower()

        for j, fd2 in enumerate(fds):
            if i >= j:
                continue

            lhs2_lower = {c.lower() for c in fd2.lhs_cols}
            rhs2_lower = fd2.rhs_col.lower()

            connected = False

            # 规则1：共享LHS列
            if lhs1_lower & lhs2_lower:
                connected = True
            # 规则2：一个FD的RHS在另一个FD的LHS中
            elif rhs1_lower in lhs2_lower or rhs2_lower in lhs1_lower:
                connected = True
            # 规则3：共享相同的RHS列
            elif rhs1_lower == rhs2_lower:
                connected = True

            if connected:
                graph[fd1].append(fd2)
                graph[fd2].append(fd1)

    return graph


def get_neighborhood(fd: FD, fd_graph: Dict[FD, List[FD]]) -> List[FD]:
    """
    返回fd的1-hop邻域，包含fd自身。

    参数：
      fd: 目标FD
      fd_graph: FD依赖图（由build_fd_graph构建）

    返回：
      [fd, neighbor1, neighbor2, ...]
    """
    return [fd] + fd_graph.get(fd, [])


def compute_lhs_degree(fd: FD, fd_graph: Dict[FD, List[FD]] = None) -> int:
    """
    返回FD的LHS列"度"——其邻域包含的FD数量（含自身）作为近似。

    如果提供了fd_graph，返回 1 + len(fd_graph.get(fd, []))。
    否则返回1。

    参数：
      fd: 目标FD
      fd_graph: 可选的FD依赖图

    返回：
      int 度数（>= 1）
    """
    if fd_graph is not None:
        return 1 + len(fd_graph.get(fd, []))
    return 1


# ============================================================
# 少数派行检测
# ============================================================

def get_minority_rows(cg: ConflictGroup, df: pd.DataFrame) -> List[int]:
    """
    返回冲突组中RHS值不等于多数值的行索引。

    参数：
      cg: 冲突组
      df: 当前数据状态

    返回：
      少数派行的索引列表；无冲突时返回空列表
    """
    if not cg.has_conflict:
        return []

    # 处理列名大小写
    col_map = {c.lower(): c for c in df.columns}
    rhs_col = cg.fd.rhs_col if cg.fd.rhs_col in df.columns else col_map.get(cg.fd.rhs_col.lower())
    if not rhs_col or rhs_col not in df.columns:
        return []

    majority = cg.majority_rhs
    minority_rows = []
    for idx in cg.row_indices:
        if idx in df.index:
            val = df.at[idx, rhs_col]
            if val != majority:
                minority_rows.append(idx)

    return minority_rows


def get_all_cg_rows(cg: ConflictGroup, df: pd.DataFrame) -> List[int]:
    """
    返回冲突组中所有行索引（不区分多数/少数）。

    与 get_minority_rows 的区别：本函数返回 CG 中的全部行，
    让 Agent 根据频率和交叉违规信号自行学习该修哪一行。

    参数：
      cg: 冲突组
      df: 当前数据状态

    返回：
      冲突组中所有有效行索引的列表；无冲突时返回空列表
    """
    if not cg.has_conflict:
        return []
    return [idx for idx in cg.row_indices if idx in df.index]


def filter_rows_for_repair(
    cg: ConflictGroup,
    df: pd.DataFrame,
    neighborhood_fds: List[FD],
) -> List[int]:
    """
    高效过滤：只返回冲突组中在至少一个邻域 FD 下是 minority 的行。

    通过批量预计算每个 FD 的 minority 行集合来避免逐行调用
    detect_row_in_conflict_groups 的 O(n*m*k) 开销。

    多数派行如果在所有邻域 FD 下都不是 minority → 已干净 → 跳过，
    不会创建不必要的 env episode。

    参数：
      cg: 冲突组
      df: 当前数据状态
      neighborhood_fds: 邻域 FD 列表（含当前 FD 自身）

    返回：
      需要在至少一个 FD 下修复的行索引列表
    """
    if not cg.has_conflict:
        return []

    valid_rows = [idx for idx in cg.row_indices if idx in df.index]
    if not valid_rows:
        return []

    col_map = {c.lower(): c for c in df.columns}

    # 预计算所有有效行的 LHS 值（每条 FD 有不同 LHS 列定义）
    # 对每条 FD，构建 "row_idx → LHS value" 的映射
    rows_by_fd_lhs: Dict[FD, Dict[int, tuple]] = {}
    for fd in neighborhood_fds:
        lhs_cols = []
        for col in fd.lhs_cols:
            if col in df.columns:
                lhs_cols.append(col)
            elif col.lower() in col_map:
                lhs_cols.append(col_map[col.lower()])
            else:
                lhs_cols = None
                break
        if lhs_cols is None:
            continue
        row_to_lhs = {}
        for idx in valid_rows:
            row_to_lhs[idx] = tuple(df.at[idx, c] for c in lhs_cols)
        rows_by_fd_lhs[fd] = row_to_lhs

    # 对每条 FD，按 LHS 分组计算 majority RHS，标记 minority 行
    rows_needing_repair: Set[int] = set()
    for fd in neighborhood_fds:
        row_to_lhs = rows_by_fd_lhs.get(fd)
        if not row_to_lhs:
            continue

        rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
        if not rhs_col or rhs_col not in df.columns:
            continue

        # 按 LHS 值分桶，收集每个桶的 RHS 计数
        lhs_buckets: Dict[tuple, Dict[Any, List[int]]] = {}
        for idx in valid_rows:
            lhs_val = row_to_lhs.get(idx)
            if lhs_val is None:
                continue
            if lhs_val not in lhs_buckets:
                lhs_buckets[lhs_val] = {}
            rhs_val = df.at[idx, rhs_col]
            lhs_buckets[lhs_val].setdefault(rhs_val, []).append(idx)

        # 对每个桶，找到 majority RHS 值，标记 minority 行
        for lhs_val, rhs_groups in lhs_buckets.items():
            if len(rhs_groups) <= 1:
                continue  # 该 LHS 组没有冲突
            # 找到 majority
            majority_rhs = max(rhs_groups, key=lambda v: len(rhs_groups[v]))
            for rhs_val, row_list in rhs_groups.items():
                if rhs_val != majority_rhs:
                    rows_needing_repair.update(row_list)

    # 保持原始 row_indices 顺序
    return [idx for idx in valid_rows if idx in rows_needing_repair]


def compute_row_conflict_counts(
    df: pd.DataFrame,
    fds: List[FD],
) -> Dict[int, int]:
    """
    计算每条元组在所有 FD 下与其他元组冲突的数量之和。

    对每条 FD：
      groupby(LHS) → 每组按 RHS 分桶 → 每行的冲突数 = 组大小 - 该行 RHS 值的出现次数

    时间复杂度 O(|FDs| × n)，n = df 行数。

    返回：{row_index: total_conflict_count}
    """
    conflict_counts: Dict[int, int] = {idx: 0 for idx in df.index}
    col_map = {c.lower(): c for c in df.columns}

    for fd in fds:
        # 解析 LHS 列
        lhs_cols = []
        for col in fd.lhs_cols:
            if col in df.columns:
                lhs_cols.append(col)
            elif col.lower() in col_map:
                lhs_cols.append(col_map[col.lower()])
            else:
                lhs_cols = None
                break
        if lhs_cols is None:
            continue

        # 解析 RHS 列
        rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
        if not rhs_col or rhs_col not in df.columns:
            continue

        # groupby(LHS) → 每组按 RHS 分桶
        grouped = df.groupby(lhs_cols, sort=False)
        for lhs_val, group in grouped:
            if len(group) <= 1:
                continue
            rhs_counts = group[rhs_col].value_counts()
            group_size = len(group)
            for idx in group.index:
                rhs_val = df.at[idx, rhs_col]
                same_rhs_count = rhs_counts.get(rhs_val, 0)
                conflict_counts[idx] += group_size - same_rhs_count

    return conflict_counts


def compute_conditional_evidence_consistency(
    df: pd.DataFrame,
    target_row: int,
    fds: List[FD],
    all_fds: List[FD],
    neighborhood_fds: Optional[List[FD]] = None,
) -> float:
    """
    条件证据一致性 (Conditional Evidence Consistency, CEC)。

    原理：
      1. 找出当前行涉及冲突的所有 FD
      2. 收集这些 FD 中涉及的所有列（LHS + RHS）
      3. 取这些 FD 的证据列的并集（外部列，不含 FD 列）
      4. 对每个证据列 E：
         - 取 E = target_row[E] 的所有行
         - 计算这些行中，FD 相关列全部等于 target_row 对应值的比例
      5. CEC = 所有证据列条件概率的平均值

    CEC 高 → 同样证据值的行 FD 列也一致 → 当前行可能是对的
    CEC 低 → 同样证据值的行 FD 列不一致 → 当前行可能是错的

    返回：[0, 1]，无证据列时返回 0.5（中立）
    """
    if target_row not in df.index:
        return 0.5

    col_map = {c.lower(): c for c in df.columns}

    # 1. 找出当前行涉及冲突的 FD
    if neighborhood_fds is not None:
        conflicting_fds = list(neighborhood_fds)
    else:
        conflicting_fds = []
        for fd in fds:
            lhs_cols = []
            for col in fd.lhs_cols:
                if col in df.columns:
                    lhs_cols.append(col)
                elif col.lower() in col_map:
                    lhs_cols.append(col_map[col.lower()])
                else:
                    lhs_cols = None
                    break
            if lhs_cols is None:
                continue
            rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
            if not rhs_col or rhs_col not in df.columns:
                continue
            # 检查该行在此 FD 下是否 minority
            group = df.groupby(lhs_cols, sort=False).get_group(
                tuple(df.at[target_row, c] for c in lhs_cols)
            )
            if len(group) > 1 and len(group[rhs_col].unique()) > 1:
                conflicting_fds.append(fd)

    if not conflicting_fds:
        return 0.5

    # 2. 收集冲突 FD 涉及的所有列
    fd_related_lower: Set[str] = set()
    for fd in conflicting_fds:
        for c in fd.lhs_cols:
            fd_related_lower.add(c.lower())
        fd_related_lower.add(fd.rhs_col.lower())

    # 3. 取证据列的并集
    evidence_cols_set: Set[str] = set()
    for fd in conflicting_fds:
        ev_cols = get_evidence_cols(fd, all_fds, list(df.columns), df=df, max_cols=3)
        for c in ev_cols:
            if c.lower() not in fd_related_lower:
                evidence_cols_set.add(c)

    if not evidence_cols_set:
        return 0.5

    # 4. 对每个证据列计算条件概率
    fd_related_cols_actual = [col_map.get(c, c) for c in fd_related_lower
                               if c in col_map or c in set(df.columns)]
    fd_related_cols_actual = [c for c in fd_related_cols_actual if c in df.columns]

    if not fd_related_cols_actual:
        return 0.5

    target_vals = {c: df.at[target_row, c] for c in fd_related_cols_actual}

    probs = []
    for ecol in evidence_cols_set:
        if ecol not in df.columns:
            continue
        e_val = df.at[target_row, ecol]
        # 过滤 E = e_val 的行
        mask = df[ecol] == e_val
        filtered = df[mask]
        if len(filtered) == 0:
            continue
        # 计数：这些行中 FD 列全部等于 target_row 对应值的比例
        match_count = 0
        for idx in filtered.index:
            if all(filtered.at[idx, c] == target_vals[c] for c in fd_related_cols_actual):
                match_count += 1
        probs.append(match_count / len(filtered))

    if not probs:
        return 0.5
    return float(np.mean(probs))


def detect_row_in_conflict_groups(
    row_idx: int,
    neighborhood_fds: List[FD],
    df: pd.DataFrame,
) -> List[Tuple[FD, ConflictGroup]]:
    """
    检测给定行在邻域FD集合中是否仍处于冲突组。

    对neighborhood_fds中的每条FD：
      1. 计算该行的LHS值
      2. 在df中找到具有相同LHS值的所有行
      3. 构建临时rhs_candidates
      4. 只要该行所在组有冲突（无论该行是多数/少数）→ 返回该CG
      5. 让模型自行判断该修哪一行（行锁级联）

    结果按LHS度降序排列（高度 = 高优先级）。

    参数：
      row_idx: 行索引
      neighborhood_fds: 邻域FD列表（包含fd自身）
      df: DataFrame

    返回：
      [(fd, cg), ...] 按LHS度降序排列
    """
    col_map = {c.lower(): c for c in df.columns}
    result: List[Tuple[FD, ConflictGroup]] = []

    for fd in neighborhood_fds:
        # 解析LHS列名（大小写容错）
        lhs_cols = []
        for col in fd.lhs_cols:
            if col in df.columns:
                lhs_cols.append(col)
            elif col.lower() in col_map:
                lhs_cols.append(col_map[col.lower()])
        if len(lhs_cols) != len(fd.lhs_cols):
            continue  # 列不存在，跳过

        # 解析RHS列名
        rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower())
        if not rhs_col or rhs_col not in df.columns:
            continue

        # 确保row_idx存在于df中
        if row_idx not in df.index:
            continue

        # 计算该行的LHS值（tuple形式）
        lhs_val = tuple(df.at[row_idx, col] for col in lhs_cols)

        # 查找所有具有相同LHS值的行
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in zip(lhs_cols, lhs_val):
            mask &= (df[col] == val)
        matching_rows = list(df[mask].index)

        if len(matching_rows) < 2:
            continue

        # 构建RHS候选值计数
        rhs_vals = df.loc[matching_rows, rhs_col].tolist()
        rhs_counter = Counter(rhs_vals)

        if len(rhs_counter) <= 1:
            continue  # 无冲突，跳过

        # 行在该冲突组中（不再区分多数/少数）→ 级联处理
        lhs_alts = _compute_lhs_alternatives(df, fd, lhs_val)
        cg = ConflictGroup(
            fd=fd,
            lhs_value=lhs_val,
            row_indices=matching_rows,
            rhs_candidates=dict(rhs_counter),
            lhs_alternatives=lhs_alts,
        )
        result.append((fd, cg))

    # 按LHS度降序排序
    def _local_lhs_degree(fd: FD) -> int:
        """在 neighborhood_fds 范围内计算近似LHS度"""
        fd_lhs_lower = {c.lower() for c in fd.lhs_cols}
        degree = 1
        for other_fd in neighborhood_fds:
            if other_fd is fd:
                continue
            other_lhs_lower = {c.lower() for c in other_fd.lhs_cols}
            if fd_lhs_lower & other_lhs_lower:
                degree += 1
        return degree

    result.sort(key=lambda x: _local_lhs_degree(x[0]), reverse=True)
    return result

