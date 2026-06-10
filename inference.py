"""
推理脚本：用训练好的模型修复脏数据
==============================
用法：
  python inference.py --dataset hospital --input data/dirty.csv --output data/repaired.csv
  python inference.py --dataset beers --input data/raw_dirty.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from fd_repair.fd_utils import (
    parse_fds, build_all_conflict_groups,
    build_fd_graph, get_neighborhood,
    compute_row_conflict_counts,
)
from fd_repair.features import FeatureExtractor
from fd_repair.policy import ActorCritic
from fd_repair.environment import RowLockRepairEnv
from config import REWARD_CONFIG, ROW_LOCK_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Inference: repair dirty data with trained RL model")
    p.add_argument("--dataset", required=True, help="Dataset name (beers, flights, hospital, soccer)")
    p.add_argument("--input", required=True, help="Dirty CSV to repair")
    p.add_argument("--gt", default=None, help="Ground truth CSV for evaluation (optional)")
    p.add_argument("--output", default=None, help="Output repaired CSV (default: input + _repaired)")
    p.add_argument("--checkpoint_dir", default="checkpoints", help="Model checkpoint dir")
    p.add_argument("--fd_dir", default="data/FD规则", help="FD rules dir")
    p.add_argument("--fd_filter", default="", help="Comma-separated FD indices to include (1-based), e.g. '1,3,5'")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ds = args.dataset

    # Load FDs
    fds_txt = os.path.join(args.fd_dir, f"{ds}_FD.txt")
    if not os.path.exists(fds_txt):
        print(f"[ERROR] FD file not found: {fds_txt}")
        return
    with open(fds_txt, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    fds_str = []
    for line in lines:
        if "|" in line:
            lhs, rhs = line.split("|", 1)
            fds_str.append(f"{lhs.strip()} -> {rhs.strip()}")
        elif "->" in line:
            fds_str.append(line)
    fds = parse_fds(fds_str)
    if ds == "hospital":
        fds = [fd for fd in fds if not (fd.lhs_cols == ["MeasureCode"] and fd.rhs_col == "Stateavg")]
    if args.fd_filter:
        indices = [int(x.strip()) - 1 for x in args.fd_filter.split(",")]
        fds = [fds[i] for i in indices if i < len(fds)]
        print(f"FD filter applied, using FDs: {[i+1 for i in indices if i < len(fds)]}")
    print(f"Loaded {len(fds)} FDs")

    fd_graph = build_fd_graph(fds)

    # Load model
    model_path = os.path.join(args.checkpoint_dir, ds, "model_final.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    ctx_shape = state_dict["context_encoder.0.weight"].shape
    feat_dim = state_dict["context_encoder.3.weight"].shape[0]
    model = ActorCritic(input_dim=ctx_shape[1], hidden_dim=ctx_shape[0], feature_dim=feat_dim).to(args.device)
    if "rhs_scorer.2.weight" in state_dict:
        old_weight = state_dict["rhs_scorer.2.weight"]
        if old_weight.shape[0] != model.rhs_scorer[-1].out_features:
            state_dict["rhs_scorer.2.weight"] = old_weight.repeat(model.rhs_scorer[-1].out_features, 1)
            state_dict["rhs_scorer.2.bias"] = state_dict["rhs_scorer.2.bias"].repeat(model.rhs_scorer[-1].out_features)
    model.load_state_dict(state_dict)
    model.eval()
    # 兼容旧模型：input_dim=37 时不传 freq 特征
    use_new_features = (model.input_dim == 38)
    print(f"Model loaded: input_dim={model.input_dim} ({'new' if use_new_features else 'old'}), "
          f"epoch={checkpoint.get('epoch','?')}, best_f1={checkpoint.get('best_f1','?')}")

    # Load data
    dirty_df = pd.read_csv(args.input)
    repaired_df = dirty_df.copy()
    print(f"Data: {len(dirty_df)} rows x {len(dirty_df.columns)} cols")

    fe = FeatureExtractor(global_df=dirty_df, fds=fds, raw_df=dirty_df)
    modified_cells = set()

    with torch.no_grad():
        conflict_counts = compute_row_conflict_counts(dirty_df, fds)
        all_cgs_flat = []
        for fd, cgs in build_all_conflict_groups(dirty_df, fds):
            for cg in cgs:
                if cg.has_conflict:
                    all_cgs_flat.append((fd, cg))

        candidate_rows = []
        seen = set()
        for fd, cg in all_cgs_flat:
            for r in cg.row_indices:
                if r not in seen:
                    seen.add(r)
                    candidate_rows.append((r, conflict_counts.get(r, 0)))
        candidate_rows.sort(key=lambda x: -x[1])

        processed_rows = set()
        for row_idx, _ in candidate_rows:
            if row_idx in processed_rows:
                continue
            processed_rows.add(row_idx)

            current_cg = None
            for fd, cgs in build_all_conflict_groups(dirty_df, fds):
                for cg in cgs:
                    if row_idx in cg.row_indices and cg.has_conflict:
                        target_fd = fd
                        current_cg = cg
                        break
                if current_cg:
                    break
            if current_cg is None:
                continue

            neighborhood = get_neighborhood(target_fd, fd_graph)
            env = RowLockRepairEnv(
                target_row=row_idx,
                initial_cg=current_cg,
                dirty_df=dirty_df,
                feature_extractor=fe,
                neighborhood_fds=neighborhood,
                clean_df=None,  # 推理时无 GT
                reward_config=REWARD_CONFIG,
                max_steps=ROW_LOCK_CONFIG["max_steps_per_episode"],
                all_fds=fds,
                use_lite=True,
            )

            obs, _ = env.reset()
            done = truncated = False
            while not (done or truncated):
                # 兼容旧模型：截断到模型期望的维度
                obs_model = obs[:model.input_dim]
                obs_t = torch.FloatTensor(obs_model).unsqueeze(0).to(args.device)
                mask = env.get_action_mask()
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(args.device)
                dist, _ = model(obs_t, mask_t)
                action = dist.probs.argmax().item()
                obs, _, done, truncated, _ = env.step(action)

            dirty_df = env.current_df
            modified_cells.update(env.modified_cells)

    repaired_df = env.current_df

    # Output
    output_path = args.output or args.input.replace(".csv", "_repaired.csv")
    repaired_df.to_csv(output_path, index=False)
    print(f"\nRepair complete: {len(modified_cells)} cells modified")
    print(f"Saved to: {output_path}")

    # Evaluation
    if args.gt:
        gt_df = pd.read_csv(args.gt)
        col_map_gt = {c.lower(): c for c in gt_df.columns}
        col_map_dirty = {c.lower(): c for c in dirty_df.columns}
        w2r = r2w = w2w = 0
        for (ridx, cname) in modified_cells:
            col_g = col_map_gt.get(cname.lower())
            if not col_g or col_g not in gt_df.columns:
                continue
            try:
                orig = str(dirty_df.at[ridx, cname])
                rep = str(repaired_df.at[ridx, cname])
                gt_val = str(gt_df.at[ridx, col_g])
                was_wrong = (orig != gt_val)
                now_correct = (rep == gt_val)
                if now_correct:
                    if was_wrong: w2r += 1
                else:
                    if was_wrong: w2w += 1
                    else: r2w += 1
            except Exception:
                pass

        # Count total errors in FD-related columns
        lhs_cols_set = set().union(*(fd.lhs_cols for fd in fds))
        rhs_cols_set = set(fd.rhs_col for fd in fds)
        fd_cols = lhs_cols_set | rhs_cols_set
        total_errors = 0
        for idx in dirty_df.index:
            if idx not in gt_df.index:
                continue
            for fd_col in fd_cols:
                col_d = col_map_dirty.get(fd_col.lower())
                col_g = col_map_gt.get(fd_col.lower())
                if col_d and col_g and col_d in dirty_df.columns and col_g in gt_df.columns:
                    try:
                        if str(dirty_df.at[idx, col_d]) != str(gt_df.at[idx, col_g]):
                            total_errors += 1
                    except Exception:
                        pass

        precision = w2r / (w2r + r2w) if (w2r + r2w) > 0 else 0.0
        recall = w2r / max(total_errors, 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\n=== Evaluation (vs GT) ===")
        print(f"  Total errors in dirty data: {total_errors}")
        print(f"  W2R (correct fix): {w2r}")
        print(f"  R2W (broke correct): {r2w}")
        print(f"  W2W (wrong→still wrong): {w2w}")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall: {recall*100:.1f}%")
        print(f"  F1: {f1:.4f}")


if __name__ == "__main__":
    main()
