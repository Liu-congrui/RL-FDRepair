"""
训练入口
========
从 prepare_dataset.py 准备好的数据中训练RL修复模型。

用法：
  # 先准备数据
  python prepare_dataset.py --dataset hospital --error_rate 0.2

  # 再训练
  python train.py --dataset hospital --epochs 50
  python train.py --dataset hospital --epochs 50 --device cpu
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fd_repair.fd_utils import parse_fds, build_all_conflict_groups, build_pseudo_clean_df
from fd_repair.features import FeatureExtractor, FEATURE_DIM
from fd_repair.policy import ActorCritic
from fd_repair.trainer import pretrain, PPOTrainer
from fd_repair.error_injection import InjectionResult, inject_errors
from config import REWARD_CONFIG, TRAIN_CONFIG, ENV_CONFIG, ERROR_INJECTION_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL FD Repair Agent")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. hospital, beers)")
    parser.add_argument("--epochs", type=int, default=100, help="PPO epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint directory. Defaults to {dataset}/checkpoints")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with pseudo_clean.csv and dirty_train.csv. "
                             "Defaults to data/prepared/{dataset}/")
    parser.add_argument("--fds_txt", type=str, default=None,
                        help="FD rules file path. Defaults to data/FD/{dataset}_FD.txt")
    parser.add_argument("--error_rate", type=float, default=0.2,
                        help="Re-inject errors at this rate if dirty_train.csv not found")
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


def load_training_data(args):
    """
    加载训练数据。

    优先从 data/prepared/{dataset}/ 读取：
      - pseudo_clean.csv → 伪GT
      - dirty_train.csv  → 脏训练数据

    若不存在，则提示先运行 prepare_dataset.py。
    """
    dataset = args.dataset
    data_dir = args.data_dir or f"data/prepared/{dataset}"
    fds_txt = args.fds_txt or f"data/FD规则/{dataset}_FD.txt"

    pseudo_clean_path = os.path.join(data_dir, "pseudo_clean.csv")
    dirty_train_path = os.path.join(data_dir, "dirty_train.csv")

    # 加载FD规则
    fds = load_fds(fds_txt)

    if os.path.exists(pseudo_clean_path) and os.path.exists(dirty_train_path):
        print(f"  Loading from {data_dir}")
        pseudo_clean = pd.read_csv(pseudo_clean_path)
        dirty_train = pd.read_csv(dirty_train_path)
        print(f"  Pseudo-clean: {pseudo_clean.shape}, Dirty-train: {dirty_train.shape}")
        # 构造 InjectionResult（dirty_df=dirty_train, clean_df=pseudo_clean）
        injection_result = InjectionResult(
            dirty_df=dirty_train,
            clean_df=pseudo_clean,
            error_records=[],
            group_labels={},
        )
    else:
        print(f"  [WARNING] Prepared data not found in {data_dir}")
        print(f"  Please run: python prepare_dataset.py --dataset {dataset}")
        print(f"  Falling back to on-the-fly injection from data/{dataset}/{dataset}.csv")

        raw_csv = f"data/{dataset}/{dataset}.csv"
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Neither prepared data nor raw CSV found: {raw_csv}")

        raw_df = pd.read_csv(raw_csv)
        pseudo_clean = build_pseudo_clean_df(raw_df, fds)
        injection_result = inject_errors(pseudo_clean, fds, seed=args.seed)
        print(f"  On-the-fly injection: {len(injection_result.error_records)} errors")

    return injection_result, fds


def format_time(seconds):
    """将秒数格式化为可读的时间字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir or f"checkpoints-new-7action/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"Training RL FD Repair Agent - {args.dataset}")
    print(f"{'='*60}")

    # 总计时开始
    total_start = time.time()

    # 加载训练数据
    print(f"\n[1] Loading training data for {args.dataset}")
    step_start = time.time()
    injection_result, fds = load_training_data(args)
    print(f"  FDs: {len(fds)}")
    print(f"  Time: {format_time(time.time() - step_start)}")

    # 显示冲突组统计
    print("\n[2] Conflict group statistics")
    step_start = time.time()
    all_cgs = build_all_conflict_groups(injection_result.dirty_df, fds)
    total_cgs = sum(len(cgs) for _, cgs in all_cgs)
    print(f"  Total conflict groups: {total_cgs}")
    print(f"  Time: {format_time(time.time() - step_start)}")

    # 初始化模型
    print(f"\n[3] Initializing model")
    step_start = time.time()
    print(f"  Feature dim: {FEATURE_DIM}")
    model = ActorCritic()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")
    print(f"  Time: {format_time(time.time() - step_start)}")

    # 监督预训练
    print(f"\n[4] Supervised Pretraining")
    step_start = time.time()
    pretrain_losses = pretrain(model, injection_result, fds, device=device)
    if pretrain_losses:
        print(f"  Final pretrain loss: {pretrain_losses[-1]:.4f}")
    print(f"  Time: {format_time(time.time() - step_start)}")

    # PPO训练
    print(f"\n[5] PPO Training ({args.epochs} epochs)")
    step_start = time.time()
    train_config = dict(TRAIN_CONFIG)
    train_config["ppo_epochs"] = args.epochs
    train_config["checkpoint_dir"] = output_dir

    # 按数据集选择 lhs_ratio
    dataset_lhs_ratios = ERROR_INJECTION_CONFIG.get("dataset_lhs_ratios", {})
    lhs_ratio = dataset_lhs_ratios.get(args.dataset, ERROR_INJECTION_CONFIG.get("lhs_ratio", 0.5))

    trainer = PPOTrainer(
        model,
        injection_result,
        fds,
        device=device,
        config=train_config,
        reward_config=REWARD_CONFIG,
        lhs_ratio=lhs_ratio,
    )
    metrics_history, best_model_state, best_epoch = trainer.train()
    print(f"  Time: {format_time(time.time() - step_start)}")

    # 加载最好的模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state["model_state_dict"])
        print(f"\n[Best Model] Loaded from epoch {best_epoch} with F1={best_model_state['f1_score']:.4f}")

    # 保存最终模型（最好的模型）
    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics_history": metrics_history,
        "feature_dim": FEATURE_DIM,
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_f1": best_model_state["f1_score"] if best_model_state else 0.0,
    }, final_path)
    print(f"[Done] Model saved to: {final_path}")

    # 总计时结束
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total training time: {format_time(total_time)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
