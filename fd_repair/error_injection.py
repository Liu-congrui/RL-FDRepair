"""
简化错误注入模块
================
只注入两种错误类型：
- 50% lhs_error：左部错误
- 50% rhs_max_cor：RHS最高频值正确（右部错误）

每种错误内部可分为：
- 替换错误（replacement）：改成另一个存在的值
- 拼写错误（typo）：改成当前值的变异版本（前缀相同、末尾改变）
"""

from __future__ import annotations
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from .fd_utils import FD


@dataclass
class ErrorRecord:
    """单个错误注入记录"""
    row_idx: int
    col: str
    original_value: Any
    injected_value: Any
    error_type: str   # "rhs" / "lhs"
    group_type: str   # "lhs_error" / "rhs_max_cor"
    error_subtype: str  # "replacement" / "typo"


@dataclass
class InjectionResult:
    """错误注入结果"""
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame
    error_records: List[ErrorRecord]
    group_labels: Dict = None

    def __post_init__(self):
        if self.group_labels is None:
            self.group_labels = {}

    def summary(self) -> str:
        type_counts: Dict[str, int] = {}
        subtype_counts: Dict[str, int] = {}
        for r in self.error_records:
            type_counts[r.group_type] = type_counts.get(r.group_type, 0) + 1
            subtype_counts[r.error_subtype] = subtype_counts.get(r.error_subtype, 0) + 1
        return (f"Total errors: {len(self.error_records)}, "
                f"by type: {type_counts}, by subtype: {subtype_counts}")


def inject_errors(
    clean_df: pd.DataFrame,
    fds: List[FD],
    seed: int = 42,
    error_rate: Optional[float] = None,
    scenario_weights: Optional[dict] = None,
    lhs_ratio: Optional[float] = None,
) -> InjectionResult:
    """
    简化错误注入：只注入 lhs_error 和 rhs_max_cor 两种。

    lhs_ratio: LHS 错误占总错误的比例（默认从 config.ERROR_INJECTION_CONFIG 读取）
    保证每行只被一个FD注错（避免多FD冲突干扰训练）
    """
    from config import ERROR_INJECTION_CONFIG
    if lhs_ratio is None:
        lhs_ratio = ERROR_INJECTION_CONFIG.get("lhs_ratio", 0.5)
    lhs_ratio = max(0.0, min(1.0, lhs_ratio))

    random.seed(seed)
    dirty_df = clean_df.copy()
    all_errors = []

    # 计算总错误数（全局）
    total_errors_to_inject = int(len(clean_df) * (error_rate or 0.2))

    # 预计算每个FD的组信息，按组数比例分配注错配额
    fd_info = []
    col_map_fd = {c.lower(): c for c in clean_df.columns}
    for fd in fds:
        groups = _find_clean_groups(clean_df, fd)
        if not groups:
            continue
        lhs_cols_fd = [c if c in clean_df.columns else col_map_fd.get(c.lower(), c)
                       for c in fd.lhs_cols]
        all_lhs_values = list({
            tuple(clean_df.at[g[0], c] for c in lhs_cols_fd if c in clean_df.columns)
            for g in groups
        })
        fd_info.append({
            'fd': fd,
            'groups': groups,
            'lhs_cols_fd': lhs_cols_fd,
            'all_lhs_values': all_lhs_values,
            'n_groups': len(groups),
        })

    if not fd_info:
        return InjectionResult(dirty_df=dirty_df, clean_df=clean_df,
                              error_records=[], group_labels={})

    total_groups = sum(d['n_groups'] for d in fd_info)

    for info in fd_info:
        fd = info['fd']
        groups = info['groups']
        lhs_cols_fd = info['lhs_cols_fd']
        all_lhs_values = info['all_lhs_values']

        # 按组数比例分配该FD的错误配额
        fd_quota = max(1, int(total_errors_to_inject * info['n_groups'] / total_groups))
        fd_lhs_target = int(fd_quota * lhs_ratio)
        fd_rhs_target = fd_quota - fd_lhs_target

        # 按行数分配：前50%的行注LHS，后50%的行注RHS
        total_rows = sum(len(g) for g in groups)
        target_lhs_rows = int(total_rows * 0.5)

        lhs_error_groups = []
        rhs_error_groups = []
        lhs_row_sum = 0

        for group_rows in groups:
            if lhs_row_sum + len(group_rows) <= target_lhs_rows:
                lhs_error_groups.append(group_rows)
                lhs_row_sum += len(group_rows)
            else:
                rhs_error_groups.append(group_rows)

        # 注入 LHS 错误
        lhs_injected = 0
        for group_rows in lhs_error_groups:
            if lhs_injected >= fd_lhs_target:
                break
            cur_lhs = tuple(clean_df.at[group_rows[0], c]
                           for c in lhs_cols_fd if c in clean_df.columns)
            other_lhs = [v for v in all_lhs_values if v != cur_lhs]
            if not other_lhs:
                continue
            errors = _inject_lhs_error(dirty_df, fd, group_rows, "lhs_error", other_lhs,
                                      target=fd_lhs_target, current=lhs_injected)
            all_errors.extend(errors)
            lhs_injected += len(errors)

        # 注入 RHS max-cor 错误
        rhs_injected = 0
        for group_rows in rhs_error_groups:
            if rhs_injected >= fd_rhs_target:
                break
            errors = _inject_rhs_max_cor(dirty_df, fd, group_rows, "rhs_max_cor",
                                        clean_df=clean_df, original_lhs_values=all_lhs_values,
                                        target=fd_rhs_target, current=rhs_injected)
            all_errors.extend(errors)
            rhs_injected += len(errors)

    return InjectionResult(
        dirty_df=dirty_df,
        clean_df=clean_df,
        error_records=all_errors,
        group_labels={},
    )


def _find_clean_groups(clean_df: pd.DataFrame, fd: FD) -> List[List[int]]:
    """找到clean数据中所有LHS相同的组（至少2行）"""
    groups = []
    col_map = {col.lower(): col for col in clean_df.columns}
    lhs_cols = []
    for col in fd.lhs_cols:
        if col in clean_df.columns:
            lhs_cols.append(col)
        elif col.lower() in col_map:
            lhs_cols.append(col_map[col.lower()])
        else:
            return []

    try:
        grouped = clean_df.groupby(lhs_cols, sort=False)
        for lhs_val, group in grouped:
            if len(group) >= 2:
                groups.append(list(group.index))
    except KeyError:
        return []

    return groups


def _generate_typo(value: Any) -> Any:
    """
    生成拼写错误。

    - 字符串：保留前80%，改变末尾
    - 数值：加减一个小偏移量
    """
    if isinstance(value, float) and pd.isna(value):
        return value
    if isinstance(value, (int, float)):
        offset = max(1, abs(int(value * 0.01)) or 1)
        return type(value)(value + random.choice([-1, 1]) * offset)

    s = str(value)
    if len(s) <= 1:
        return s + "0"

    split_idx = max(1, int(len(s) * 0.8))
    prefix = s[:split_idx]

    op = random.choice(['delete', 'replace', 'add'])
    if op == 'delete' and len(s) > split_idx:
        return prefix
    elif op == 'replace':
        if s[-1].isdigit():
            new_char = str((int(s[-1]) + 1) % 10)
        else:
            new_char = chr(ord(s[-1]) + 1) if s[-1] != 'z' else 'a'
        return prefix + new_char
    else:
        return prefix + "0"


def _inject_lhs_error(
    dirty_df: pd.DataFrame,
    fd: FD,
    group_rows: List[int],
    group_type: str,
    other_lhs_values: Optional[List[tuple]] = None,
    target: int = None,
    current: int = None,
) -> List[ErrorRecord]:
    """
    注入 LHS 错误：从该组中选行，改其 LHS 值为其他真实存在的 LHS 值。

    只注入替换错误（改成其他真实存在的LHS值），不注入拼写错误。
    只用频率 > 3 的 LHS 值作为替换候选。

    参数：
      target: 总目标错误数
      current: 当前已注入的错误数
    """
    errors = []
    size = len(group_rows)

    # 计算这个组最多能注多少个错误
    if target is not None and current is not None:
        remaining = target - current
        if remaining <= 0:
            return errors
        # 这个组最多注 remaining 个，但不超过 30-50% 的行
        max_errors = min(remaining, max(1, int(size * random.uniform(0.3, 0.5))))
    else:
        max_errors = max(1, int(size * random.uniform(0.3, 0.5)))

    error_rows = random.sample(group_rows, min(max_errors, size))

    col_map = {c.lower(): c for c in dirty_df.columns}
    lhs_cols = [c if c in dirty_df.columns else col_map.get(c.lower(), c) for c in fd.lhs_cols]

    if not other_lhs_values:
        return errors

    # 获取 RHS 列（用于冲突检查和多数值占比检查）
    col_map_rhs = {c.lower(): c for c in dirty_df.columns}
    rhs_col = fd.rhs_col if fd.rhs_col in dirty_df.columns else col_map_rhs.get(fd.rhs_col.lower())

    # 计算每个LHS值的频率，只保留频率 > 3 的
    lhs_freq = {}
    for lhs_val in other_lhs_values:
        mask = None
        for i, col in enumerate(lhs_cols):
            if col in dirty_df.columns:
                col_mask = (dirty_df[col] == lhs_val[i]) if i < len(lhs_val) else pd.Series([False] * len(dirty_df))
                mask = col_mask if mask is None else (mask & col_mask)

        if mask is not None:
            count = mask.sum()
            if count > 3:
                lhs_freq[lhs_val] = count

    # 只用频率 > 3 的LHS值作为候选
    high_freq_candidates = list(lhs_freq.keys())

    # 预计算每个候选LHS组的RHS列表（用于冲突检查和多数值占比检查）
    candidate_rhs_list_map = {}
    if rhs_col and rhs_col in dirty_df.columns:
        for cand_lhs in high_freq_candidates:
            mask = pd.Series([True] * len(dirty_df), index=dirty_df.index)
            for col, val in zip(lhs_cols, cand_lhs):
                if col in dirty_df.columns:
                    mask &= (dirty_df[col] == val)
            candidate_rhs_list_map[cand_lhs] = dirty_df.loc[mask, rhs_col].tolist()

    MIN_MAJORITY_RATIO = 0.6  # 注入后目标组多数值占比不得低于此值

    for row_idx in error_rows:
        current_lhs = tuple(dirty_df.at[row_idx, c] for c in lhs_cols if c in dirty_df.columns)
        row_rhs = dirty_df.at[row_idx, rhs_col] if rhs_col and rhs_col in dirty_df.columns else None

        candidates = [v for v in high_freq_candidates if v != current_lhs]
        if not candidates:
            continue

        if row_rhs is not None and candidate_rhs_list_map:
            # 过滤候选：注入后目标组多数值占比 >= 60%，且能产生冲突
            safe_conflict_candidates = []
            for cand_lhs in candidates:
                if cand_lhs not in candidate_rhs_list_map:
                    continue
                target_rhs_list = candidate_rhs_list_map[cand_lhs]
                if not target_rhs_list:
                    continue
                # 模拟注入后的RHS列表
                simulated = target_rhs_list + [row_rhs]
                counts = Counter(simulated)
                majority_count = counts.most_common(1)[0][1]
                majority_ratio = majority_count / len(simulated)
                # 必须产生冲突（row_rhs不在目标组）且多数值占比 >= 60%
                if row_rhs not in set(target_rhs_list) and majority_ratio >= MIN_MAJORITY_RATIO:
                    safe_conflict_candidates.append(cand_lhs)

            if safe_conflict_candidates:
                candidates = safe_conflict_candidates
            else:
                # fallback：只保证多数值占比 >= 60%，不强求产生冲突
                safe_candidates = []
                for cand_lhs in candidates:
                    if cand_lhs not in candidate_rhs_list_map:
                        continue
                    target_rhs_list = candidate_rhs_list_map[cand_lhs]
                    if not target_rhs_list:
                        continue
                    simulated = target_rhs_list + [row_rhs]
                    counts = Counter(simulated)
                    majority_ratio = counts.most_common(1)[0][1] / len(simulated)
                    if majority_ratio >= MIN_MAJORITY_RATIO:
                        safe_candidates.append(cand_lhs)
                if safe_candidates:
                    candidates = safe_candidates
                # 若仍无满足条件的候选，跳过此行不注错
                else:
                    continue

        target_lhs = random.choice(candidates)
        subtype = "replacement"

        # 写入错误值
        for i, col in enumerate(lhs_cols):
            if col not in dirty_df.columns:
                continue
            original = dirty_df.at[row_idx, col]
            injected = target_lhs[i] if i < len(target_lhs) else str(original) + "_err"
            dirty_df.at[row_idx, col] = injected
            errors.append(ErrorRecord(row_idx, col, original, injected, "lhs", group_type, subtype))

    return errors


def _inject_rhs_max_cor(
    dirty_df: pd.DataFrame,
    fd: FD,
    group_rows: List[int],
    group_type: str,
    clean_df: Optional[pd.DataFrame] = None,
    original_lhs_values: Optional[List[tuple]] = None,
    target: int = None,
    current: int = None,
) -> List[ErrorRecord]:
    """
    注入 RHS max-cor 错误：选择最高频的RHS值作为正确值，其余行改成虚假值。

    只有当 majority_val == correct_val 时才注入错误（保证 max-cor 语义）。

    错误方式：
    - 50% replacement：改成数据集中存在的其他 RHS 值
    - 50% typo：改成当前值的拼写错误版本

    参数：
      target: 总目标错误数
      current: 当前已注入的错误数
    """
    errors = []
    size = len(group_rows)

    col_map = {c.lower(): c for c in dirty_df.columns}
    rhs_col = fd.rhs_col if fd.rhs_col in dirty_df.columns else col_map.get(fd.rhs_col.lower(), fd.rhs_col)

    if rhs_col not in dirty_df.columns:
        return errors

    # 获取正确的RHS值
    correct_val = None
    if clean_df is not None and original_lhs_values:
        col_map_clean = {c.lower(): c for c in clean_df.columns}
        rhs_col_clean = rhs_col if rhs_col in clean_df.columns else col_map_clean.get(rhs_col.lower(), rhs_col)
        lhs_cols_clean = [col if col in clean_df.columns else col_map_clean.get(col.lower(), col) for col in fd.lhs_cols]

        # 用当前组的实际 LHS 值查 clean_df
        lhs_cols_dirty = [c if c in dirty_df.columns else col_map.get(c.lower(), c)
                          for c in fd.lhs_cols]
        lhs_val = tuple(dirty_df.at[group_rows[0], c]
                        for c in lhs_cols_dirty if c in dirty_df.columns)
        mask = pd.Series([True] * len(clean_df), index=clean_df.index)
        for col, val in zip(lhs_cols_clean, lhs_val):
            if col in clean_df.columns:
                mask &= (clean_df[col] == val)

        clean_matched = clean_df[mask]
        if len(clean_matched) > 0 and rhs_col_clean in clean_df.columns:
            correct_rhs = clean_matched[rhs_col_clean].mode()
            if len(correct_rhs) > 0:
                correct_val = correct_rhs[0]

    # 统计该组RHS值频率
    rhs_vals = [dirty_df.at[r, rhs_col] for r in group_rows]
    rhs_counter = Counter(rhs_vals)
    majority_val = rhs_counter.most_common(1)[0][0]

    # 计算这个组最多能注多少个错误
    if target is not None and current is not None:
        remaining = target - current
        if remaining <= 0:
            return errors
        max_errors = remaining
    else:
        max_errors = float('inf')

    # 限制同一组内错误比例不超过30%，保证正确值始终是绝对多数
    max_by_ratio = max(1, int(size * 0.3))

    # 如果全是最高频值，选不超过30%的行改成其他值
    non_majority_rows = [r for r in group_rows if dirty_df.at[r, rhs_col] != majority_val]
    if not non_majority_rows:
        n_errors = max(1, int(size * random.uniform(0.2, 0.3)))
        n_errors = min(n_errors, int(max_errors), max_by_ratio, size - 1)
        error_rows = random.sample(group_rows, n_errors)
    else:
        n_errors = max(1, int(len(non_majority_rows) * random.uniform(0.5, 0.7)))
        n_errors = min(n_errors, int(max_errors), max_by_ratio)
        error_rows = random.sample(non_majority_rows, min(n_errors, len(non_majority_rows)))

    # 其他RHS值（用于replacement）
    ref_val = correct_val if correct_val is not None else majority_val
    other_rhs_vals = [v for v in dirty_df[rhs_col].unique() if v != ref_val]

    existing_rhs_vals = set(dirty_df[rhs_col].unique())

    for row_idx in error_rows:
        original = dirty_df.at[row_idx, rhs_col]

        use_replacement = random.random() < 0.5
        if use_replacement and other_rhs_vals:
            injected = random.choice(other_rhs_vals)
            subtype = "replacement"
        else:
            typo_val = _generate_typo(original)
            # 若 typo 生成了数据集中不存在的唯一值，改用 replacement 避免产生无法修复的孤立错误
            if typo_val not in existing_rhs_vals and other_rhs_vals:
                injected = random.choice(other_rhs_vals)
                subtype = "replacement"
            else:
                injected = typo_val
                subtype = "typo"

        col_dtype = dirty_df[rhs_col].dtype
        if col_dtype != object:
            try:
                injected = col_dtype.type(injected)
            except (ValueError, TypeError):
                pass
        dirty_df.at[row_idx, rhs_col] = injected
        errors.append(ErrorRecord(row_idx, rhs_col, original, injected, "rhs", group_type, subtype))

    return errors
