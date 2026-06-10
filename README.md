# RL-FDRepair

Functional Dependency Violation Repair via Sequential Decision Making.

An FD violation repair system powered by PPO reinforcement learning. It models FD conflict repair as a Markov Decision Process, uses evidence-based features to identify error types and repair targets, and employs a row-lock cascade mechanism to handle cross-FD cascading repairs.

---

## Project Structure

```
├── config.py               # Global configuration (reward, training, network)
├── train.py                # Training entry point
├── inference.py            # Inference / repair entry point
├── prepare_dataset.py      # Data preparation (pseudo_clean + dirty_train)
├── requirements.txt
├── fd_repair/
│   ├── __init__.py
│   ├── environment.py      # RL environment (RowLockRepairEnv: row-lock cascade)
│   ├── features.py         # Feature extractor (37 dim)
│   ├── policy.py           # PPO Actor-Critic (37 -> 128 -> 64, LayerNorm + ReLU)
│   ├── trainer.py          # Supervised pretraining + PPO + curriculum learning
│   ├── fd_utils.py         # FD parsing, conflict groups, FD graph, evidence ranking
│   ├── error_injection.py  # Error injection (RHS / LHS / majority-mislead / multi-cell)
│   └── evaluator.py        # Evaluation metrics
└── data/
    ├── FD规则/             # FD definitions for 4 datasets
    ├── inject_errors/      # Dirty data ({dataset}-new/: clean + dirty 0.05~0.30)
    ├── change_LHS_rate/    # LHS ratio sensitivity experiment data
    └── prepared/           # Training-ready data (pseudo_clean + dirty_train)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch >= 2.0`, `numpy >= 1.24`, `pandas >= 2.0`

---

## Quick Start

### 1. Prepare Data

```bash
python prepare_dataset.py \
  --dataset hospital \
  --input_csv data/inject_errors/hospital-new/hospital_dirty_0.20_m.csv \
  --error_rate 0.2
```

Outputs `data/prepared/{dataset}/pseudo_clean.csv` and `dirty_train.csv`.

### 2. Train

```bash
python train.py --dataset hospital --epochs 100
python train.py --dataset beers --epochs 50 --device cpu
```

Model saved to `checkpoints/{dataset}/model_final.pt`.

### 3. Inference / Repair

```bash
# Basic repair
python inference.py \
  --dataset hospital \
  --input data/inject_errors/hospital-new/hospital_dirty_0.20_m.csv \
  --output data/repaired.csv

# With ground-truth evaluation
python inference.py \
  --dataset hospital \
  --input data/inject_errors/hospital-new/hospital_dirty_0.20_m.csv \
  --gt data/inject_errors/hospital-new/hospital_clean.csv
```

---

## MDP Formulation

| Element | Definition |
|---------|------------|
| **State** | 37-dim feature vector: group(8) + RHS(4) + evidence(4) + LHS(15) + row-lock(4) + target-row(2) |
| **Action** | RHS unification (3) + LHS reassignment (3) + NO_OP (1) = 7 discrete actions |
| **Reward** | cell-level repair correctness + row-level detection reward + shaping penalties |
| **Episode** | Locks one tuple per episode, tracks cascades via the FD dependency graph, stops when the tuple has no conflicts or after K_max = 20 steps |

### Action Space

```
Action 0~2 : RHS unify — pick top-3 candidate values
Action 3~5 : LHS reassign — pick top-3 evidence-ranked alternative LHS groups
Action 6   : NO_OP  — current tuple is correct, do nothing
```

### Reward Formula

```
r_t = r_t^cell + r_t^row + r_t^shape

  r_t^cell:   wrong -> right  +5.0  (w_+)
              right -> wrong  -10.0 (w_-, w_- > w_+)
              wrong -> wrong  -1.0

  r_t^row:    correctly repair dirty row  +20.0  (R_repair)
              correctly skip clean row      0.0  (R_skip)
              wrongly edit clean row       -3.0  (P_edit)
              miss a dirty row            -20.0  (P_miss)

  r_t^shape:  invalid action  -0.5  (beta = 0.5)
              repeat action   -0.3  (gamma = 0.3)
              step cost       -0.03 (alpha)
              terminal bonus  +5.0  (R_term = 5.0, complete repair)
```

---

## Training Pipeline

1. **Supervised Pretraining** (30 epochs): Injects synthetic errors into pseudo-clean data to construct D_train, generates oracle action labels row by row, and pretrains via cross-entropy for a stable initial policy.
2. **PPO Fine-tuning** (curriculum learning): Phase 1 (first 20 epochs) trains only on injected-error rows to learn "fix wrong rows = good"; Phase 2 (next 80 epochs) mixes in up to 50% clean rows to learn "skip correct rows". GAE (lambda = 0.95) traces delayed conflicts across cascades. PPO hyperparameters: clip epsilon = 0.1, gamma = 0.99, lr = 5e-5.

---

## Datasets

| Dataset | # Rows | # FDs | # Cols |
|---------|--------|-------|--------|
| flights | 2,000 | 4 | 5 |
| soccer | 1,000 | 3 | 10 |
| hospital | 1,000 | 14 | 20 |
| beers | 2,410 | 3 | 11 |

---

## Configuration

All parameters in `config.py`:

| Block | Key Parameters |
|-------|---------------|
| `REWARD_CONFIG` | w_+ = 5.0, w_- = -10.0, R_repair = 20.0, P_edit = -3.0, P_miss = -20.0, beta = 0.5, gamma = 0.3, alpha = -0.03, R_term = 5.0 |
| `TRAIN_CONFIG` | pretrain 30 epochs (lr 5e-4), PPO 30 epochs (lr 5e-5), clip epsilon = 0.1, gamma = 0.99, lambda = 0.95 |
| `ENV_CONFIG` | max_steps_per_episode = 5, max_candidates K = S = 3 |
| `NETWORK_CONFIG` | feature_dim = 64, hidden_dim = 128 |
| `ERROR_INJECTION_CONFIG` | LHS error ratio = 0.5 (reduced to 0.2 for datasets where LHS columns span multiple FDs, e.g. cars, tax1) |
| `ROW_LOCK_CONFIG` | cascade max steps K_max = 20 |
| `INFERENCE_CONFIG` | confidence margin threshold = 0.05 |
