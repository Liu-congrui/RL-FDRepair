"""
评估指标模块
===========
计算训练/验证/测试阶段的各类评估指标。

训练/验证阶段（有GT）：
  - cell-level accuracy
  - precision / recall / F1（以被修改的错误单元格为正例）
  - wrong2right, right2wrong, wrong2wrong, right2right
  - FD violation reduction

测试阶段（无GT）：
  - FD冲突减少量
  - 修改单元格数量
  - 每个冲突组的决策日志
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .fd_utils import FD, count_fd_violations, count_all_fd_violations


@dataclass
class EvalMetrics:
    """评估指标容器"""

    # 单元格级别（有GT时）
    total_cells: int = 0
    modified_cells: int = 0
    wrong2right: int = 0   # 错误被正确修复
    right2wrong: int = 0   # 正确被改错
    wrong2wrong: int = 0   # 错误改成另一个错误
    right2right: int = 0   # 正确保持正确（没改）

    # FD冲突（始终有）
    initial_violations: Dict[str, int] = field(default_factory=dict)
    final_violations: Dict[str, int] = field(default_factory=dict)

    @property
    def cell_accuracy(self) -> float:
        """修复后的单元格级准确率"""
        if self.total_cells == 0:
            return 1.0
        correct = self.wrong2right + self.right2right
        return correct / self.total_cells

    @property
    def precision(self) -> float:
        """在所有修改中，正确修复的比例"""
        total_modified = self.wrong2right + self.right2wrong + self.wrong2wrong
        if total_modified == 0:
            return 1.0
        return self.wrong2right / total_modified

    @property
    def recall(self) -> float:
        """所有错误单元格中，被正确修复的比例"""
        total_errors = self.wrong2right + self.wrong2wrong + self.right2wrong
        # 注意：total_errors应该是原始dirty中的错误数
        # 这里用wrong2right + wrong2wrong作为分母（已处理的错误）
        total_wrong = self.wrong2right + self.wrong2wrong
        if total_wrong == 0:
            return 1.0 if self.wrong2right == 0 else 0.0
        return self.wrong2right / max(total_wrong, 1)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def violation_reduction(self) -> Dict[str, int]:
        """每条FD的冲突减少量"""
        return {
            fd_str: self.initial_violations.get(fd_str, 0) - self.final_violations.get(fd_str, 0)
            for fd_str in self.initial_violations
        }

    @property
    def total_violation_reduction(self) -> int:
        return sum(self.violation_reduction.values())

    def __str__(self) -> str:
        lines = ["=" * 50, "Evaluation Metrics", "=" * 50]
        if self.total_cells > 0:
            lines += [
                f"Cell Accuracy  : {self.cell_accuracy:.4f}",
                f"Precision      : {self.precision:.4f}",
                f"Recall         : {self.recall:.4f}",
                f"F1             : {self.f1:.4f}",
                f"Wrong->Right   : {self.wrong2right}",
                f"Right->Wrong   : {self.right2wrong}",
                f"Wrong->Wrong   : {self.wrong2wrong}",
                f"Right->Right   : {self.right2right}",
                f"Modified Cells : {self.modified_cells}",
            ]
        lines += [
            f"Violation Reduction: {self.violation_reduction}",
            f"Total Reduction : {self.total_violation_reduction}",
        ]
        return "\n".join(lines)


def compute_metrics(
    original_dirty_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    fds: List[FD],
    clean_df: Optional[pd.DataFrame] = None,
    dirty_cells: Optional[Set[Tuple[int, str]]] = None,
) -> EvalMetrics:
    """
    计算完整评估指标。

    参数：
      original_dirty_df : 修复前的dirty数据
      repaired_df       : 修复后的数据
      fds               : FD列表
      clean_df          : 干净数据（有GT时提供）
      dirty_cells       : 被注入错误的单元格集合 {(row_idx, col), ...}

    返回：
      EvalMetrics对象
    """
    metrics = EvalMetrics()

    # FD冲突统计
    fd_str_map = {repr(fd): fd for fd in fds}
    for fd in fds:
        fd_str = repr(fd)
        metrics.initial_violations[fd_str] = count_fd_violations(original_dirty_df, fd)
        metrics.final_violations[fd_str] = count_fd_violations(repaired_df, fd)

    if clean_df is None:
        return metrics

    # 单元格级别比较
    all_cols = list(repaired_df.columns)
    all_indices = list(repaired_df.index)

    for idx in all_indices:
        for col in all_cols:
            metrics.total_cells += 1

            dirty_val = original_dirty_df.at[idx, col]
            repaired_val = repaired_df.at[idx, col]
            clean_val = clean_df.at[idx, col]

            is_originally_wrong = (dirty_val != clean_val)
            is_now_correct = (repaired_val == clean_val)
            was_modified = (dirty_val != repaired_val)

            if was_modified:
                metrics.modified_cells += 1

            if is_originally_wrong and is_now_correct:
                metrics.wrong2right += 1
            elif not is_originally_wrong and not is_now_correct:
                metrics.right2wrong += 1
            elif is_originally_wrong and not is_now_correct:
                metrics.wrong2wrong += 1
            else:  # not is_originally_wrong and is_now_correct
                metrics.right2right += 1

    return metrics

 # 用于训练时的实时监控，计算单个episode（冲突组）的指标
def compute_episode_metrics(
    initial_cg_violations: int,
    final_cg_violations: int,
    modified_cells: Set[Tuple[int, str]],
    clean_df: Optional[pd.DataFrame],
    repaired_df: pd.DataFrame,
    original_dirty_df: pd.DataFrame,
) -> Dict:
    """
    计算单个episode（冲突组）的指标。
    用于训练时的实时监控。
    """
    result = {
        "initial_violations": initial_cg_violations,
        "final_violations": final_cg_violations,
        "violation_reduction": initial_cg_violations - final_cg_violations,
        "modified_cells": len(modified_cells),
    }

    if clean_df is not None:
        w2r = r2w = w2w = 0
        for (idx, col) in modified_cells:
            dirty_val = original_dirty_df.at[idx, col]
            repaired_val = repaired_df.at[idx, col]
            clean_val = clean_df.at[idx, col]

            orig_wrong = (dirty_val != clean_val)
            now_correct = (repaired_val == clean_val)

            if orig_wrong and now_correct:
                w2r += 1
            elif not orig_wrong and not now_correct:
                r2w += 1
            elif orig_wrong and not now_correct:
                w2w += 1

        result.update({
            "wrong2right": w2r,
            "right2wrong": r2w,
            "wrong2wrong": w2w,
        })

    return result
