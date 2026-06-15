# 题目二实验分析

> 数值来自 `outputs/report_B.json` 与 `outputs/report_ABC_FAIR.json`（checkpoint 080000）。

## 1. 实验设置

| 项目 | 配置 |
|------|------|
| 策略 | LeRobot ACT（`chunk_size=100`，`n_action_steps=10`） |
| 随机种子 | `SEED=42` |
| 单环境训练 | `calvin_env_B`（80k 步） |
| 混合训练 | `calvin_env_ABC` → `train_abc_fair`（80k 步） |
| Zero-shot 评测 | `calvin_env_D`（267{,}373 帧） |

### 指标说明

| 指标 | 含义 | 报告用途 |
|------|------|----------|
| `action_l1_loss` | 离线 per-frame 动作 L1（归一化空间） | 跨环境主指标 |
| `generalization_gap.action_l1_loss.absolute` | `L1_D - L1_train` | **视觉分布偏移量化** |
| `chunk_step_l1` | chunk 内第 k 步平均 L1 | **Action Chunking 鲁棒性** |
| `chunk_horizon_degradation` | 末步 L1 / 首步 L1（>1 表示远期更难） | Chunk 退化分析 |
| `action_dim_l1` | 7 维动作各维度 L1 | 哪类动作偏移最大 |
| `action_l1_median` / `p90` | 典型帧 / 最差 10% | 鲁棒性补充 |

## 2. 一键生成评测报告

```bash
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair
```

输出：`report_*.json`、`chunk_l1_*.png`、`gap_comparison.png`。

## 3. 训练曲线

| 实验 | WandB run | 输出目录 |
|------|-----------|----------|
| B | `act_calvin_b` | `outputs/train_b/` |
| ABC\_fair | `act_calvin_abc_fair` | `outputs/train_abc_fair/` |

报告插图：`task2/wandb_train_L1loss.png`、`wandb_train_update_s.png`。

### 观察

- 两套实验均在约 2k 步内从 loss $\sim$9 快速下降至 $<$1，80k 步时 B 为 0.100、ABC\_fair 为 0.126。
- 相同 80k 步下，B 约覆盖 4.77 epoch，ABC\_fair 仅约 1.59 epoch（混合集约为 B 的 3 倍大），故 ABC\_fair 保留略高训练 loss 并不意外。
- `train/update_s` 在 B 上偶发尖峰更高，混合训练反而更平稳。

## 4. 视觉分布偏移 + Action Chunking

### 表 1：泛化 Gap

| 训练集 | In-dist L1 | Zero-shot L1 (D) | Gap (absolute) | Gap (relative) |
|--------|-----------|------------------|----------------|----------------|
| B | 0.310 | 0.522 | 0.212 | 1.68$\times$ |
| ABC\_fair | 0.390 | 0.463 | 0.073 | 1.19$\times$ |

### 表 2：分位数（D zero-shot）

| 训练集 | median | $p_{90}$ |
|--------|--------|----------|
| B | 0.457 | 0.885 |
| ABC\_fair | 0.411 | 0.769 |

### 表 3：Action Chunking 退化（$k{=}0$ vs $k{=}64$）

| 训练集 | 首步 L1 (in-dist) | 末步 L1 (in-dist) | deg. (in-dist) | deg. (D) |
|--------|-------------------|-------------------|----------------|----------|
| B | 0.341 | 0.509 | 1.49 | 1.51 |
| ABC\_fair | 0.389 | 0.616 | 1.58 | 1.57 |

最大逐步间隙：B 在 $k{=}56$ 为 0.297；ABC\_fair 在 $k{=}50$ 为 0.103。

### 图

- WandB：`wandb_train_L1loss.png`、`wandb_train_update_s.png`
- 评测：`chunk_l1_B.png`、`chunk_l1_ABC_FAIR.png`、`gap_comparison.png`

### 分析要点

- **视觉偏移**：B 的 Gap 约为 ABC\_fair 的 2.9 倍；混合训练牺牲 in-dist 精度换取跨环境鲁棒性。
- **Action Chunking**：degradation $\approx$1.5 在两套模型上几乎不变，偏移主要抬升整条 chunk 曲线。
- **动作维度**：D 上 rot\_x / rot\_y 最差；ABC\_fair 在旋转与夹爪维均优于 B。
- **鲁棒性**：ABC\_fair 的 median 与 $p_{90}$ 均低于 B，说明改善来自典型帧与长尾帧的双重抑制。

## 5. 结论

在相同 80k 优化预算下，**ABC\_fair 混合训练显著降低对未见环境 D 的泛化间隙**，且 Action Chunking 结构在偏移下仍保持可预测的远期退化模式。局限：未接入 CALVIN 仿真 Success Rate；未对 A/C 单环境做完整消融。
