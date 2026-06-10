# RL-FDRepair

Functional Dependency Violation Repair via Sequential Decision Making.

基于 PPO 强化学习的函数依赖（FD）冲突数据修复系统。将 FD 冲突修复建模为 Markov Decision Process，通过 evidence-based 特征判断错误类型与修复目标，利用 row-lock cascade 机制处理跨 FD 的级联修复。

---

## 项目结构

```
├── config.py               # 全局配置
├── train.py                # 训练入口
├── inference.py            # 推理 / 修复入口
├── prepare_dataset.py      # 数据准备
├── requirements.txt
├── fd_repair/
│   ├── fd_utils.py         # FD 解析、冲突组构建、FD 依赖图、证据排名
│   ├── error_injection.py  # 错误注入（RHS / LHS / majority-mislead / multi-cell）
│   ├── environment.py      # RL 环境（FDRepairEnv + RowLockRepairEnv）
│   ├── features.py         # 特征提取器（37 维：group + evidence + RHS + LHS + row-lock + target-row）
│   ├── policy.py           # PPO Actor-Critic（37 → 128 → 64, LayerNorm + ReLU）
│   ├── trainer.py          # 监督预训练 + PPO 训练器（课程学习）
│   └── evaluator.py        # 评估指标
└── data/
    ├── FD规则/             # 8 个数据集的 FD 定义
    ├── inject_errors/      # 脏数据（7 个数据集 × 多个错误率）
    └── prepared/           # 训练数据（pseudo_clean + dirty_train）
```

---

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch >= 2.0`, `numpy >= 1.24`, `pandas >= 2.0`

---

## 快速开始

### 1. 准备数据

```bash
python prepare_dataset.py \
  --dataset hospital \
  --input_csv data/inject_errors/hospital-new/hospital_dirty_0.20_m.csv \
  --error_rate 0.2
```

输出 `data/prepared/{dataset}/pseudo_clean.csv` 和 `dirty_train.csv`。

### 2. 训练

```bash
python train.py --dataset hospital --epochs 100
python train.py --dataset beers --epochs 50 --device cpu
```

模型保存至 `checkpoints-new-7action/{dataset}/model_final.pt`。

### 3. 推理

```bash
python inference.py \
  --dataset hospital \
  --input data/dirty.csv \
  --output data/repaired.csv

# 带 GT 评估
python inference.py \
  --dataset hospital \
  --input data/dirty.csv \
  --gt data/ground_truth.csv
```

---

## MDP 形式化

| 元素 | 定义 |
|------|------|
| **State** | 37 维特征向量：group(8) + evidence(4) + RHS(4) + LHS(15) + row-lock(4) + target-row(2) |
| **Action** | RHS 统一 (3) + LHS 移出 (3) + NO_OP (1) = 7 个离散动作 |
| **Reward** | cell-level 修复正确性 + row-level detection 奖励 + shaping 惩罚 |
| **Episode** | 逐行锁定，通过 FD 依赖图级联追踪，直至该行无冲突或达 K_max = 20 步 |

### 动作空间

```
动作 0~2 : RHS 统一 — 选 top-3 候选值
动作 3~5 : LHS 移出 — 选 top-3 evidence-ranked 替代 LHS 组
动作 6   : NO_OP  — 当前行正确，不修改
```

### Reward 公式

```
r_t = r_t^cell + r_t^row + r_t^shape

  r_t^cell:   wrong → right  +5.0  (w_+)
              right → wrong  -10.0 (w_-, w_- > w_+)
              wrong → wrong  -1.0

  r_t^row:    correctly repair dirty row  +20.0  (R_repair)
              correctly skip clean row      0.0  (R_skip)
              wrongly edit clean row       -3.0  (P_edit)
              miss a dirty row            -20.0  (P_miss)

  r_t^shape:  invalid action  -0.5  (β = 0.5)
              repeat action   -0.3  (γ = 0.3)
              step cost       -0.03 (α)
              terminal bonus  +5.0  (R_term = 5.0, complete repair)
```

---

## 训练流程

1. **监督预训练**：在 D_pseudo 注入合成错误构造 D_train，逐行生成最优动作标签，交叉熵预训练 30 epochs。
2. **PPO 微调**：课程学习 — Phase 1 (20 epochs) 只训练注错行；Phase 2 (80 epochs) 混入最多 50% 正确行。GAE (λ = 0.95) 追溯级联中的延迟冲突。每组超参：clip ε = 0.1, γ = 0.99。

---

## 支持的数据集

| 数据集 | # Rows | # FDs | # Cols |
|--------|--------|-------|--------|
| flights | 2,000 | 4 | 5 |
| soccer | 1,000 | 3 | 10 |
| hospital | 1,000 | 14 | 20 |
| beers | 2,410 | 3 | 11 |

---

## 配置

所有参数在 `config.py` 中：

| 配置块 | 内容 |
|--------|------|
| `REWARD_CONFIG` | cell-level 奖励权重、selection 奖励、shaping 惩罚 |
| `TRAIN_CONFIG` | 预训练 & PPO 超参、课程学习、error rate 范围 |
| `ENV_CONFIG` | max steps per episode, max candidates (K = S = 3) |
| `NETWORK_CONFIG` | feature_dim = 64, hidden_dim = 128 |
| `ERROR_INJECTION_CONFIG` | LHS error ratio (default 0.5) |
| `ROW_LOCK_CONFIG` | cascade max steps = 20 |
| `INFERENCE_CONFIG` | 置信度门控阈值 |
