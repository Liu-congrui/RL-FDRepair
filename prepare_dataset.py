"""
数据集准备脚本
==============
从脏数据集中提取伪GT，再注入错误生成训练数据。

伪GT提取逻辑：
  对每个FD的每个冲突组（LHS相同但RHS不同的元组），
  用多数RHS值替换少数RHS值，得到修复后的伪GT。

用法：
  python prepare_dataset.py --dataset hospital --error_rate 0.2
  python prepare_dataset.py --dataset beers --error_rate 0.15

输出：
  data/prepared/{dataset}/pseudo_clean.csv  ← 伪GT
  data/prepared/{dataset}/dirty_train.csv   ← 训练用脏数据
"""

import argparse
import os
import sys
import pandas as pd
from collections import Counter
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fd_repair.fd_utils import parse_fds, build_conflict_groups, FD
from fd_repair.error_injection import inject_errors


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for RL training")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. hospital, beers)")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input dirty CSV (e.g. data/inject_errors/hospital/hospital_dirty_0.3_m.csv)")
    parser.add_argument("--fds_txt", type=str, default=None,
                        help="Path to FD rules file. Defaults to data/FD规则/{dataset}_FD.txt")
    parser.add_argument("--error_rate", type=float, default=0.5,
                        help="Error injection rate for training data (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to data/prepared-new/{dataset}/")
    return parser.parse_args()


def load_fds(fds_txt: str):
    fds_str = []
    with open(fds_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                lhs, rhs = line.split("|", 1)
                fds_str.append(f"{lhs.strip()} -> {rhs.strip()}")
            elif "->" in line:
                fds_str.append(line)
    return parse_fds(fds_str)


def extract_pseudo_clean(df: pd.DataFrame, fds: List[FD]) -> pd.DataFrame:
    """
    从脏数据中提取伪GT：用每个FD冲突组中的多数RHS值替换少数RHS值。

    逻辑：
      对每个FD的每个冲突组（LHS相同但RHS不同）：
        - 找到多数RHS值（出现次数最多的值）
        - 将所有少数RHS值替换为多数值
        - 无冲突的组保持不变

    例：[1,a]×10, [1,b]×1, [1,c]×1
        → 多数值为 'a'，将 [1,b] 和 [1,c] 都改为 [1,a]
    """
    pseudo_clean = df.copy()
    col_map = {c.lower(): c for c in df.columns}

    for fd in fds:
        lhs_cols = [c if c in df.columns else col_map.get(c.lower(), c) for c in fd.lhs_cols]
        rhs_col = fd.rhs_col if fd.rhs_col in df.columns else col_map.get(fd.rhs_col.lower(), fd.rhs_col)

        if not all(c in df.columns for c in lhs_cols) or rhs_col not in df.columns:
            continue

        try:
            grouped = pseudo_clean.groupby(lhs_cols, sort=False)
        except KeyError:
            continue

        for _, group in grouped:
            rhs_counter = Counter(group[rhs_col].tolist())

            if len(rhs_counter) <= 1:
                continue

            # 找到多数值
            majority_val, _ = rhs_counter.most_common(1)[0]

            # 找出少数值所在的行，替换为多数值
            if pd.isna(majority_val):
                minority_mask = ~group[rhs_col].isna()
            else:
                minority_mask = group[rhs_col].isna() | (group[rhs_col] != majority_val)

            minority_indices = group.index[minority_mask]
            if len(minority_indices) > 0:
                pseudo_clean.loc[minority_indices, rhs_col] = majority_val

    return pseudo_clean


def main():
    args = parse_args()

    dataset = args.dataset
    fds_txt = args.fds_txt or f"data/FD规则/{dataset}_FD.txt"
    output_dir = args.output_dir or f"data/prepared/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    # 直接使用用户指定的输入数据集
    input_csv = args.input_csv

    print(f"\n{'='*60}")
    print(f"Preparing dataset: {dataset}")
    print(f"{'='*60}")

    print(f"\n[1] Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"    Shape: {df.shape}")

    print(f"\n[2] Loading FD rules from {fds_txt}")
    fds = load_fds(fds_txt)
    print(f"    FDs loaded: {len(fds)}")
    for fd in fds:
        print(f"      {fd}")

    # 从脏数据中提取伪GT：保留冲突组中多数值的"多余"行
    # 每个冲突组每种RHS值只留1行作代表，其余多数值行进入pseudo_clean
    print(f"\n[3] Extracting pseudo-clean from dirty data (majority rows)")
    pseudo_clean = extract_pseudo_clean(df, fds)
    print(f"    Pseudo-clean rows: {len(pseudo_clean)} / {len(df)}")

    if len(pseudo_clean) < 10:
        print(f"    [WARNING] Too few pseudo-clean rows ({len(pseudo_clean)}), "
              f"falling back to full input data.")
        pseudo_clean = df.copy()

    pseudo_clean_path = os.path.join(output_dir, "pseudo_clean.csv")
    pseudo_clean.to_csv(pseudo_clean_path, index=False)
    print(f"    Saved: {pseudo_clean_path}")

    # 在 pseudo_clean 上注入错误生成训练数据
    print(f"\n[4] Injecting errors (error_rate={args.error_rate}, seed={args.seed})")
    injection_result = inject_errors(pseudo_clean, fds, seed=args.seed, error_rate=args.error_rate)
    dirty_train = injection_result.dirty_df
    print(f"    Errors injected: {len(injection_result.error_records)}")
    print(f"    {injection_result.summary()}")

    dirty_train_path = os.path.join(output_dir, "dirty_train.csv")
    dirty_train.to_csv(dirty_train_path, index=False)
    print(f"    Saved: {dirty_train_path}")

    # 统计冲突组
    from fd_repair.fd_utils import build_all_conflict_groups
    all_cgs = build_all_conflict_groups(dirty_train, fds)
    total_cgs = sum(len(cgs) for _, cgs in all_cgs)
    print(f"\n[5] Conflict groups in dirty_train: {total_cgs}")

    print(f"\nDone. Training data ready in: {output_dir}")
    print(f"  Pseudo-GT:   {pseudo_clean_path}")
    print(f"  Dirty train: {dirty_train_path}")


if __name__ == "__main__":
    main()

