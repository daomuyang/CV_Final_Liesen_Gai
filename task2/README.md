# 题目二：LeRobot ACT × CALVIN 跨环境泛化

基于 LeRobot **ACT**，完成 B 单环境训练、ABC 混合训练（80k 步）与 D 环境 Zero-shot 离线评测。

**说明**：虚拟环境（`.venv` / conda）未随仓库上传，需在本地按下文重建。数据集与最优模型权重见百度网盘（链接与提取码见 `report_task2.pdf`）。

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
├── grad_norm.csv             # WandB 导出的 train/grad_norm 数据（B 与 ABC_fair）
├── kld_loss.csv              # WandB 导出的 train/kld_loss 数据（B 与 ABC_fair）
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

## 主要操作
```bash
### 1. 环境（GPU 服务器）
bash scripts/setup_server.sh
conda activate cv_hw3_task2

### 2. 数据：从网盘解压五个 calvin_env_* 到 data/（见下文）
python scripts/verify_dataset.py

### 3. 权重：从网盘解压到 outputs/（见下文「最优模型」）
若只有 080000，需为评测脚本创建 last 软链接：
ln -sfn 080000 outputs/train_b/checkpoints/last
ln -sfn 080000 outputs/train_abc_fair/checkpoints/last

### 4. 测试（无需重新训练）
export SEED=42
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair

### 5. 重新训练（如需要）
bash scripts/train.sh B server
bash scripts/train.sh ABC_fair server
```

macOS 本地仅做冒烟：`bash scripts/setup_local.sh && bash scripts/run_smoke.sh`

---

## 最优模型：放置位置与格式

网盘下载后，**必须保持以下目录结构**（仅 `model.safetensors` 无法评测）。注意：请在保留GitHub上下载的outputs文件夹中内容的基础上，加入网盘上下载的outputs/train_b和outputs/train_abc_fair文件夹，请勿直接覆盖GitHub上的outputs文件夹。

```
task2/outputs/
├── train_b/checkpoints/
│   ├── 080000/                          # 最优步
│   │   └── pretrained_model/            # ★ 完整目录，7 个文件
│   └── last/                            # 评测脚本读取此目录（内容与 080000 相同，需为评测脚本创建 last 软链接，具体方式见“主要操作”中的指令）
│       └── pretrained_model/
└── train_abc_fair/checkpoints/
    ├── 080000/
    │   └── pretrained_model/
    └── last/
        └── pretrained_model/
```

每个 `pretrained_model/` 目录需包含 **全部 7 个文件**（LeRobot 0.5.1 格式）：

| 文件 | 作用 |
|------|------|
| `model.safetensors` | ACT 策略权重（~197 MB） |
| `config.json` | 网络结构（chunk_size=100 等） |
| `train_config.json` | 训练配置 |
| `policy_preprocessor.json` | 输入归一化流水线 |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | 归一化统计量 |
| `policy_postprocessor.json` | 输出反归一化流水线 |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | 反归一化统计量 |

验证权重是否就绪：

```bash
ls outputs/train_b/checkpoints/last/pretrained_model/model.safetensors
ls outputs/train_abc_fair/checkpoints/last/pretrained_model/model.safetensors
```

---

## 环境配置

| 项目 | 版本 |
|------|-----|
| Python | 3.12 |
| PyTorch | 2.10.0 + CUDA 12.1（服务器） |
| LeRobot | 0.5.1 |
| 随机种子 | `SEED=42` |

### GPU 服务器（conda）

```bash
cd task2
bash scripts/setup_server.sh    # 读取 environment.yml，创建 cv_hw3_task2
conda activate cv_hw3_task2
```

### GPU 服务器（venv）

```bash
cd task2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-server.txt
export HF_HUB_OFFLINE=1
```

### macOS 本地冒烟

```bash
bash scripts/setup_local.sh && source .venv/bin/activate
```

验证：`python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__)"`

所有 `scripts/*.sh` 会自动激活 `.venv` 或 conda 环境 `cv_hw3_task2`。

---

## 数据准备

将网盘中的五个目录解压到 `task2/data/`，**目录名不可改**：

| 目录 | Episodes |
|------|----------|
| `data/calvin_env_A` | 4468 |
| `data/calvin_env_B` | 4468 |
| `data/calvin_env_C` | 4467 |
| `data/calvin_env_ABC` | 13403 |
| `data/calvin_env_D` | 4467 |

格式：LeRobot v3.0（`meta/info.json` + parquet + AV1 mp4）。

```bash
python scripts/verify_dataset.py   # 全部 [OK] 后再训练/评测
```

---

## 训练（Train）

```bash
cd task2
export SEED=42

bash scripts/train.sh B server          # outputs/train_b/
bash scripts/train.sh ABC_fair server   # outputs/train_abc_fair/（ABC 为别名）
```

每 10k 步保存 checkpoint；80k 结束后 `checkpoints/last` 自动更新。若目录已存在需先 `rm -rf outputs/train_b`。

本地冒烟（20 步）：`bash scripts/run_smoke.sh`

---

## 测试（Test）

```bash
# 完整评测 + 报告 JSON/图表
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair

# 仅 zero-shot → D
bash scripts/eval_zeroshot.sh B server
bash scripts/eval_zeroshot.sh ABC_fair server

# 快速自检（50 batch）
bash scripts/diagnose.sh B
bash scripts/diagnose.sh ABC_fair
```

主要输出：`outputs/eval_*_on_*.json`、`outputs/report_*.json`、`outputs/gap_comparison.png`、`outputs/chunk_l1_*.png`

---

## 常见问题

**`No checkpoint under .../checkpoints/last`**  
从网盘恢复权重后执行：`ln -sfn 080000 outputs/train_*/checkpoints/last`

**`Output exists: outputs/train_b`**  
重新训练前：`rm -rf outputs/train_b`

**`ModuleNotFoundError: lerobot`**  
重新执行环境配置并 `conda activate cv_hw3_task2` 或 `source .venv/bin/activate`
