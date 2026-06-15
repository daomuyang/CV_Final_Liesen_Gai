# 实验配置对照表

## 已完成的训练实验

| 实验名（CLI） | 输出目录 | 训练数据 | steps | 说明 |
|---------------|----------|----------|-------|------|
| `B` | `outputs/train_b` | `calvin_env_B` | **80000** | 作业第 1 步：单环境 B 基线 |
| `ABC_fair`（`ABC` 为别名） | `outputs/train_abc_fair` | `calvin_env_ABC` | **80000** | 作业第 2–3 步：A/B/C 混合训练，与 B **相同优化步数** |

> CLI 中 `ABC` 与 `ABC_fair` 均映射到 `outputs/train_abc_fair`（80k），便于脚本兼容。

## 超参数一致性（B vs ABC_fair）

以下参数在 `scripts/train.sh` 中**完全一致**：

| 超参数 | B | ABC_fair |
|--------|---|----------|
| `policy.type` | act | act |
| `chunk_size` | 100 | 100 |
| `n_action_steps` | 10 | 10 |
| `batch_size`（CUDA） | 16 | 16 |
| `num_workers`（CUDA） | 8 | 8 |
| `lr`（ACT 默认） | 1e-5 | 1e-5 |
| `optimizer` | AdamW preset | AdamW preset |
| `save_freq` | 10000 | 10000 |
| `log_freq` | 100 | 100 |
| `tolerance_s` | 0.02 | 0.02 |
| `eval_freq` | 0 | 0 |
| **steps** | **80000** | **80000** |
| **seed** | **42** | **42** |

## 运行命令

```bash
# 作业主实验
bash scripts/train.sh B server
bash scripts/train.sh ABC_fair server   # 或 bash scripts/train.sh ABC server

# 评测
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair
```

## 评测结果文件

| 实验 | 离线结果 |
|------|----------|
| B | `outputs/report_B.json`、`outputs/eval_B_on_D.json` |
| ABC_fair | `outputs/report_ABC_FAIR.json`、`outputs/eval_ABC_FAIR_on_D.json` |
