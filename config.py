REWARD_CONFIG = {
      # 修复奖励（cell-level）：w_- > w_+ (惩罚 > 奖励，避免级联破坏)
      "wrong_to_right": 5.0,     # w_+
      "right_to_wrong": -10.0,   # w_- (paper: w_- > w_+)
      "wrong_to_wrong": -1.0,
      "right_to_right": 0.0,

      "conflict_reduce": 0.0,
      "conflict_increase": 0.0,

      "global_viol_reduce": 0.0,
      "global_viol_increase": 0.0,

      "invalid_action": -0.5,    # β (paper: β = 0.5)
      "repeat_action": -0.3,     # γ (paper: γ = 0.3)
      "no_op_penalty": 0.0,

      "step_cost": -0.03,        # α (step penalty)

      "terminal_correct_bonus": 5.0,   # R_term (paper: R_term = 5.0)
      "terminal_partial_bonus": 0.0,
      "terminal_fail_penalty": 0.0,

      # Row-level detection reward（paper: r_t^row）
      "correct_selection_bonus": 20.0,   # R_repair: 修对错行
      "wrong_selection_penalty": -3.0,   # P_edit: 修错正确行
      "correct_skip_bonus": 0.0,         # R_skip: 跳过正确行
      "missed_repair_penalty": -20.0,    # P_miss: 跳过错行
}

SAFETY_CONFIG = {
      "enable_safety_check": False,
      "max_global_viol_increase": 10,
      "enable_fallback": False,
      "confidence_threshold": 0.1,
      "fallback_strategy": "majority",
}

ENV_CONFIG = {
      "max_steps_per_episode": 5,
      "max_candidates": 3,       # K = S = 3 (paper)
}

NETWORK_CONFIG = {
      "feature_dim": 64,
      "hidden_dim": 128,
}

ERROR_INJECTION_CONFIG = {
    # LHS 错误占总错误的比例，剩余为 RHS 错误
    # beers/hospital/soccer: 0.5（LHS 列不跨 FD，eject 副作用小）
    # cars: 0.2（car_name 同时是 FD1/FD2 的 LHS，eject 会破坏其他 FD）
    # tax1: 0.2（zip 同时是 FD1/FD2 的 LHS，eject 会破坏其他 FD）
    "lhs_ratio": 0.5,  # 默认值
    # 按数据集覆盖（LHS 列跨多个 FD 时降低）
    "dataset_lhs_ratios": {
        "cars": 0.2,
        "tax1": 0.2,
    },
}

TRAIN_CONFIG = {
      "pretrain_epochs": 30,
      "pretrain_lr": 5e-4,
      "pretrain_batch_size": 32,

      "ppo_epochs": 30,
      "ppo_steps_per_epoch": 512,
      "ppo_mini_batch_size": 64,
      "ppo_update_iters": 4,
      "ppo_lr": 5e-5,
      "ppo_clip_epsilon": 0.1,
      "ppo_gamma": 0.99,
      "ppo_lam": 0.95,
      "ppo_vf_coef": 0.5,
      "ppo_entropy_coef": 0.1,   # 提高探索，让NO_OP有机会被采样
      "ppo_entropy_anneal": True,
      "ppo_entropy_anneal_rate": 0.99,  # 更慢的退火速度
      "ppo_min_entropy": 0.02,    # 防止熵坍缩到0，保持最低探索
      "ppo_max_grad_norm": 0.5,

      # 每个 epoch 随机从这些 error_rate 中选一个重新注错
      # 覆盖所有测试集的难度范围（0.05~0.3）
      "train_error_rates": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],

      # 课程学习：前 N 个 epoch 只用注错行训练，避免冷启动时探索不足
      "curriculum_error_only_epochs": 20,    # Phase 1: 只训练注错行，学会"修错行=好"
      "curriculum_mix_epochs": 80,           # Phase 2: 混入正确行（最多50%），学会"跳过正确行"
      "curriculum_mix_correct_ratio": 0.5,   # Phase 2 正确行上限50%，保持训练数据平衡
      "curriculum_never_all": True,          # 永远不进入"全量数据"阶段，防止策略坍缩

      "save_interval": 10,
      "log_interval": 5,
      "checkpoint_dir": "checkpoints",  # train.py 会自动改为 {dataset}/checkpoints
}

INFERENCE_CONFIG = {
      "enable_confidence_gate": True,
      "confidence_margin_thresh": 0.05,
      "majority_ratio_thresh": 0.5,  # 改为0.5
}

ROW_LOCK_CONFIG = {
    "max_steps_per_episode": 20,
}
