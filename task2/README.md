# 题目二：LeRobot ACT × CALVIN 跨环境泛化

基于 [LeRobot](https://github.com/huggingface/lerobot) **ACT** 策略，在 CALVIN 风格数据上完成：

1. **B 单环境训练**（基线）
2. **ABC 混合训练**（`ABC_fair`，与 B 对齐 80k 步）
3. **D 环境 Zero-shot 离线评测**（跨视觉域泛化）

本仓库**不包含**可上传的虚拟环境目录（`.venv` / conda env 需在每台机器上自行创建）。请严格按下文「环境配置」重建环境后再运行命令。

---

## 目录结构

```
task2/
├── README.md                 # 本文档
├── environment.yml           # GPU 服务器 conda 环境
├── environment-local.yml     # macOS 本地冒烟 conda 环境
├── environment-server.yml    # 同 environment.yml
├── requirements-server.txt   # 服务器 pip 依赖（lerobot 等）
├── requirements-local.txt    # 本地 pip 依赖
├── data/                     # 数据集（见「数据准备」）
│   ├── calvin_env_A/
│   ├── calvin_env_B/
│   ├── calvin_env_C/
│   ├── calvin_env_ABC/
│   └── calvin_env_D/
├── scripts/                  # 训练 / 评测脚本
│   ├── setup_server.sh       # 服务器一键装环境
│   ├── setup_local.sh        # macOS 本地装环境
│   ├── verify_dataset.py     # 数据校验
│   ├── train.sh              # 训练
│   ├── eval_report.sh        # 完整评测报告
│   ├── eval_offline.sh       # 单次离线评测
│   ├── eval_zeroshot.sh      # Zero-shot → D
│   ├── eval_compare.sh       # B vs ABC_fair 对比图
│   ├── diagnose.sh           # 训练后快速自检
│   └── run_smoke.sh          # 本地端到端冒烟
└── outputs/                  # 训练与评测产物
    ├── train_b/
    ├── train_abc_fair/
    ├── eval_*_on_*.json
    ├── report_*.json
    └── logs/
```

---

## 环境配置

### 重要说明

| 项目 | 说明 |
|------|------|
| **未上传** | `.venv/`、conda 环境目录、`wandb/` 缓存 |
| **已上传** | 代码、`requirements-*.txt`、`environment*.yml`、训练好的 `outputs/.../checkpoints/080000/` |
| **核心版本** | Python 3.12 · PyTorch 2.10.0 · LeRobot 0.5.1 |
| **随机种子** | 默认 `SEED=42`（`export SEED=42`） |

### 方式 A：GPU 服务器

**硬件**：NVIDIA GPU + CUDA 12.1+

**一键安装（有 conda）**：

```bash
cd task2
bash scripts/setup_server.sh
conda activate cv_hw3_task2
```

`setup_server.sh` 会读取 `environment.yml` 创建 conda 环境 `cv_hw3_task2`，并自动运行数据校验。

**手动安装（无 conda，使用 venv + pip）**：

```bash
cd task2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-server.txt
export HF_HUB_OFFLINE=1
python scripts/verify_dataset.py
```

**WandB（训练日志，离线模式）**：

```bash
export WANDB_API_KEY=<your_key>   
wandb login                       

# 训练完成后同步离线日志
wandb sync outputs/train_b/wandb/offline-run-*
wandb sync outputs/train_abc_fair/wandb/offline-run-*
```

### 方式 B：macOS 本地（冒烟测试，不跑完整训练）

用于验证数据、脚本与评测流程。完整 80k 训练请在 GPU 服务器执行。

**一键安装**：

```bash
cd task2
bash scripts/setup_local.sh
source .venv/bin/activate
```

**或使用 conda**：

```bash
cd task2
conda env create -f environment-local.yml
conda activate cv_hw3_task2_local
python scripts/verify_dataset.py
```

本地会自动使用 **MPS**（Apple Silicon）或 **CPU**；视频解码后端为 `pyav`。

### 环境验证

无论哪种方式，安装后应看到类似输出：

```bash
python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__)"
# lerobot 0.5.1 | torch 2.10.0

python scripts/verify_dataset.py
# [OK] calvin_env_A ... calvin_env_D
# All datasets passed verification.
```

### 激活环境的规则

所有 `scripts/*.sh` 会自动尝试：

1. `task2/.venv`（venv 方式）
2. conda 环境 `cv_hw3_task2`（服务器）

因此服务器用 conda、本地用 venv 均可，**无需每次手动 activate**（脚本内部会处理）。

---

## 数据准备

### 目录要求

将 LeRobot v3.0 格式数据放入 `task2/data/`，**目录名必须完全一致**：

| 目录 | 用途 | Episodes |
|------|------|----------|
| `data/calvin_env_A` | 单环境 A（混合训练源） | 4468 |
| `data/calvin_env_B` | 单环境 B 训练 | 4468 |
| `data/calvin_env_C` | 单环境 C（混合训练源） | 4467 |
| `data/calvin_env_ABC` | A+B+C 联合训练 | 13403 |
| `data/calvin_env_D` | Zero-shot 评测 | 4467 |

每个目录需包含：

```
calvin_env_B/
├── meta/
│   ├── info.json          # 必须：episodes、features、fps 等
│   └── stats.json
├── data/                  # parquet 帧数据
└── videos/                # AV1 mp4 视频
    ├── observation.images.image/
    └── observation.images.wrist_image/
```

格式：**LeRobot v3.0**（`codebase_version: v3.0`，parquet + AV1 mp4）。

### 获取数据

1. 从百度网盘下载五个 `calvin_env_*` 目录（链接和提取码见报告），并放置到 `task2/data/`，保持目录名不变
2. 运行校验：
```bash
cd task2
python scripts/verify_dataset.py
```

全部显示 `[OK]` 后方可训练。

---

## 训练（Train）

### 作业主实验

在 **GPU 服务器**上执行（每条约数小时，视 GPU 而定）：

```bash
cd task2
export SEED=42

# 实验 1：单环境 B 基线（80k 步）
bash scripts/train.sh B server

# 实验 2：ABC 混合训练（80k 步，与 B 对齐）
bash scripts/train.sh ABC_fair server
```

`ABC` 与 `ABC_fair` 等价，均映射到 `outputs/train_abc_fair/`。

### 训练超参数（B 与 ABC_fair 完全一致）

| 参数 | 值 |
|------|-----|
| `policy.type` | act |
| `chunk_size` | 100 |
| `n_action_steps` | 10 |
| `batch_size`（CUDA） | 16 |
| `num_workers`（CUDA） | 8 |
| `lr` | 1e-5（AdamW） |
| `steps` | 80000 |
| `save_freq` | 10000 |
| `seed` | 42 |

### 训练产物

```
outputs/train_b/checkpoints/
├── 010000/ ... 080000/     # 每 10k 步保存
└── last/                   # 最新步（80k 训练结束后 = 080000）

outputs/train_b/checkpoints/080000/pretrained_model/
├── model.safetensors       # ★ 策略权重（提交核心文件）
├── config.json
├── train_config.json
├── policy_preprocessor.json
└── policy_postprocessor.json
```

**提交的最优权重**（两套实验各一份）：

- `outputs/train_b/checkpoints/080000/pretrained_model/`
- `outputs/train_abc_fair/checkpoints/080000/pretrained_model/`

### 本地冒烟训练

```bash
cd task2
bash scripts/run_smoke.sh
```

仅训练 20 步 × 3 个 episode，约 2–5 分钟，用于确认环境无误。

---

## 测试 / 评测（Test）

### 1. 训练后快速自检（50 batch，约 1–2 分钟）

```bash
bash scripts/diagnose.sh B
bash scripts/diagnose.sh ABC_fair
```

检查 in-dist L1 是否足够低、zero-shot D 是否高于 in-dist（跨环境偏移属正常）。

### 2. 完整评测报告

```bash
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair
```

**输出文件**：

| 文件 | 内容 |
|------|------|
| `outputs/eval_B_on_B.json` | B 模型在 B 环境 in-dist L1 |
| `outputs/eval_B_on_D.json` | B 模型在 D 环境 zero-shot L1 |
| `outputs/report_B.json` | Gap + Action Chunking 汇总 |
| `outputs/chunk_l1_B.png` | Chunk 逐步 L1 曲线 |
| `outputs/gap_comparison.png` | B vs ABC_fair 泛化间隙对比 |

ABC_fair 对应 `eval_ABC_FAIR_on_*`、`report_ABC_FAIR.json`、`chunk_l1_ABC_FAIR.png`。

### 3. 仅 Zero-shot 评测（D 环境）

```bash
bash scripts/eval_zeroshot.sh B server
bash scripts/eval_zeroshot.sh ABC_fair server
```

### 4. 任意环境离线评测

```bash
# 语法：bash scripts/eval_offline.sh <训练实验> <评测环境> [smoke|server]
bash scripts/eval_offline.sh ABC_fair D server    # zero-shot
bash scripts/eval_offline.sh ABC_fair ABC server  # in-distribution
```

`smoke` 模式只跑 3 个 episode、5 个 batch，适合本地快速验证：

```bash
bash scripts/eval_offline.sh B D smoke
```

---

## 结果解读

### 主要指标

| 指标 | 含义 |
|------|------|
| `action_l1_loss` | 离线 per-frame 动作 L1（归一化空间），**跨环境主指标** |
| `generalization_gap.action_l1_loss.absolute` | `L1_D − L1_train`，量化视觉分布偏移 |
| `chunk_step_l1` | chunk 内第 k 步平均 L1，衡量 Action Chunking 鲁棒性 |
| `chunk_horizon_degradation` | 末步 L1 / 首步 L1（>1 表示远期更难） |
| `success_rate_t0_05` 等 | 极严 per-frame 阈值，**不是** CALVIN 任务成功率 |

### 参考数值（checkpoint 080000）

| 训练集 | In-dist L1 | Zero-shot L1 (D) | Gap |
|--------|-----------|------------------|-----|
| B | 0.310 | 0.522 | 0.212 |
| ABC_fair | 0.390 | 0.463 | 0.073 |

- B 在训练环境拟合更好，但跨到 D 时泛化间隙更大
- ABC_fair 牺牲部分 in-dist 精度，显著降低对 D 的泛化间隙
- 训练 loss：B 80k 约 0.100，ABC_fair 约 0.126（混合数据更难拟合，属预期）

---

## 常见问题

### `ModuleNotFoundError: No module named 'lerobot'`

环境未激活或未安装。重新执行「环境配置」一节，然后 `python -c "import lerobot"` 验证。

### `Output exists: outputs/train_b`

完整训练不允许覆盖已有目录。删除或改名旧目录后再训：

```bash
rm -rf outputs/train_b
bash scripts/train.sh B server
```

### `No checkpoint under .../checkpoints/last`

训练未完成或 `pretrained_model/` 缺失。确认 `model.safetensors` 存在。

### 在 Mac 上评测 CUDA 训练的 checkpoint

评测脚本已自动将 `device_processor` 覆盖为当前设备（MPS/CPU），可直接运行 `eval_offline.sh ... smoke`。

### `nvidia-smi` 不可用 / 训练很慢

确认在 GPU 服务器执行 `bash scripts/train.sh ... server`，且 `detect_device` 返回 `cuda`。

---

## 完整复现流程

**GPU 服务器从头到尾**：

```bash
cd task2

# 1. 环境
bash scripts/setup_server.sh
conda activate cv_hw3_task2

# 2. 数据（放置到 task2/data/ 后）
python scripts/verify_dataset.py

# 3. 训练
export SEED=42
bash scripts/train.sh B server
bash scripts/train.sh ABC_fair server

# 4. 评测
bash scripts/diagnose.sh B
bash scripts/diagnose.sh ABC_fair
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair

# 5. 同步 WandB
wandb sync outputs/train_b/wandb/offline-run-*
wandb sync outputs/train_abc_fair/wandb/offline-run-*
```

**macOS 本地冒烟**：

```bash
cd task2
bash scripts/setup_local.sh
source .venv/bin/activate
bash scripts/run_smoke.sh
```
