# 题目二：LeRobot ACT × CALVIN 跨环境泛化

基于 [LeRobot](https://github.com/huggingface/lerobot) **ACT**，在 CALVIN 风格数据上完成 **B 单环境训练**、**ABC 混合训练（ABC\_fair，80k 步）** 与 **D 环境 Zero-shot 评测**。

## 数据集（本地 `data/`，勿改名）

| 目录 | 用途 | Episodes |
|------|------|----------|
| `data/calvin_env_A` | 单环境 A（混合训练源） | 4468 |
| `data/calvin_env_B` | 单环境 B 训练 | 4468 |
| `data/calvin_env_C` | 单环境 C（混合训练源） | 4467 |
| `data/calvin_env_ABC` | A+B+C 联合训练 | 13403 |
| `data/calvin_env_D` | Zero-shot 评测 | 4467 |

格式：LeRobot v3.0（parquet + AV1 mp4）。

## 环境配置

### 本地（macOS 冒烟）

```bash
cd task2
bash scripts/setup_local.sh
source .venv/bin/activate
bash scripts/run_smoke.sh   # 约 2–5 分钟
```

### 阿里云 GPU 服务器

```bash
cd task2
bash scripts/setup_server.sh
source .venv/bin/activate

export WANDB_API_KEY=<your_key>
wandb login

# 可复现：默认 SEED=42
export SEED=42

# B 基线
bash scripts/train.sh B server

# ABC 混合（80k 步，与 B 对齐）
bash scripts/train.sh ABC_fair server

# 训练后自检
bash scripts/diagnose.sh B
bash scripts/diagnose.sh ABC_fair

# 完整评测报告（Gap + Action Chunking 曲线）
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair

wandb sync outputs/train_*/wandb/offline-run-*
```

## 可复现性

| 变量 | 默认 | 说明 |
|------|------|------|
| `SEED` | `42` | 传给 `lerobot-train --seed` 与 eval |
| `CUDNN_DETERMINISTIC` | `true` | 略慢，但更稳定 |

## 命令速查

| 任务 | 命令 |
|------|------|
| 验证数据 | `python scripts/verify_dataset.py` |
| 训练 B / ABC\_fair | `SEED=42 bash scripts/train.sh <B\|ABC_fair> server` |
| 快速自检 | `bash scripts/diagnose.sh <B\|ABC_fair>` |
| **完整报告（推荐）** | `bash scripts/eval_report.sh <B\|ABC_fair> server` |
| B vs ABC\_fair 对比图 | `bash scripts/eval_compare.sh B ABC_fair` |
| Zero-shot → D | `bash scripts/eval_zeroshot.sh <B\|ABC_fair> server` |
| 任意离线评测 | `bash scripts/eval_offline.sh ABC_fair D server` |
| 仿真 Success Rate | `docs/calvin_sim_setup.md` |
| 报告模板 | `docs/analysis.md` |

## 结果解读

- **训练 loss**：B 在 80k 步约 0.100，ABC\_fair 约 0.126（混合数据更难拟合）。
- **zero-shot `action_l1_loss`**：跨环境 D 的离线 L1 高于训练 loss 是预期行为。
- **`success_rate_t0_05`**：阈值 0.05 极严，不是 CALVIN 任务成功率。
- **交作业**：WandB 训练曲线截图 + 离线 `action_l1_loss` 对比 + Action Chunking 分析。

## 输出目录

```
outputs/
├── train_b/
├── train_abc_fair/
├── eval_*_on_*.json
├── report_*.json          # Gap + chunk 分析汇总
├── chunk_l1_*.png         # Action Chunking 曲线
├── gap_comparison.png     # B vs ABC_fair 对比
├── diagnose_*_*.json
└── logs/
wandb_train_*.png          # WandB 对比截图（报告用）
docs/analysis.md
```

## 作业提交清单

- [x] B WandB loss 曲线（`wandb_train_L1loss.png`）
- [x] ABC\_fair WandB loss 曲线（同上对比图）
- [x] `eval_*_on_D.json`（离线 L1 对比）
- [x] `report_task2.pdf`（NeurIPS 风格报告）
- [ ] checkpoints 云盘链接（若要求）
