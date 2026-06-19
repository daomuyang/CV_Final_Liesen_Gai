# 计算机视觉课程期末项目

本仓库包含两道题目：

| 题目 | 目录 | 内容 |
|------|------|------|
| **题目一** | [`task1/`](task1/) | 多物体 3D 重建与 AIGC 场景融合（COLMAP、2DGS、threestudio SDS、Magic123、Blender） |
| **题目二** | [`task2/`](task2/) | LeRobot ACT × CALVIN 跨环境泛化（B 单环境 / ABC 混合训练，D 环境 Zero-shot 评测） |

详细说明见各任务子目录 README：[`task1/README.md`](task1/README.md)、[`task2/README.md`](task2/README.md)。

---

## 仓库结构

```
cv_final/
├── README.md                    # 本文档
├── report.tex                   # 课程报告tex文件
├── task1/                       # 题目一代码与日志
│   ├── environment.yml
│   ├── scripts/
|   ├── README.md                    
│   └── outputs/
└── task2/                       # 题目二代码与评测产物
    ├── environment.yml
    ├── environment-server.yml
    ├── environment-local.yml
    ├── requirements-server.txt
    ├── requirements-local.txt
    ├── README.md    
    ├── scripts/
    ├── data/
    └── outputs/
```

---

## 数据与权重（百度网盘）

题目一、题目二的数据与模型权重均在同一份网盘中：

- 链接：https://pan.baidu.com/s/1dlkCUwM8b2_L2cgbFi3RKA?pwd=stq2  
- 提取码：`stq2`

| 网盘内容 | 解压目标 | 说明 |
|----------|----------|------|
| `submission_assets/` | `task1/submission_assets/` | 题目一全部输入与输出（约 8.8 GB） |
| `third_party/` | `task1/third_party/` | 2DGS、COLMAP、Magic123、threestudio（约 200 MB） |
| `calvin_env_*`（5 个目录） | `task2/data/` | 题目二 LeRobot v3.0 数据集 |
| `outputs/train_b/`、`outputs/train_abc_fair/` | `task2/outputs/` | 题目二最优 checkpoint（**勿覆盖** GitHub 上已有评测 JSON） |

GitHub 仓库：https://github.com/daomuyang/CV_Final_Liesen_Gai/tree/main

---

# 题目一：多物体 3D 重建与场景融合

## 环境配置

### 依赖概览

| 项目 | 值 |
|------|-----|
| Conda 环境名 | `task1` |
| Python | 3.8 |
| PyTorch | 2.0.1 + CUDA 11.8 |
| 配置文件 | [`task1/environment.yml`](task1/environment.yml) |

主要组件：COLMAP、2D Gaussian Splatting、threestudio（DreamFusion SDS）、Magic123、Open3D、diffusers、xformers 等（完整列表见 `environment.yml` 的 `pip:` 段）。

### GPU 服务器

```bash
cd task1
bash scripts/setup_server.sh
conda activate task1
```

`setup_server.sh` 会读取 `environment.yml` 创建/激活 `task1`，并对 `third_party/2d-gaussian-splatting` 应用 alpha-mask 补丁。

### 手动创建 Conda 环境

```bash
cd task1
conda env create -f environment.yml -n task1
conda activate task1
```

> **注意**：题目一依赖 `third_party/` 中的第三方库，需先从网盘解压到 `task1/third_party/`。流水线通过 `scripts/common.sh` 中的 `DATASET_ROOT` 与 `THIRD_PARTY_DIR` 读写路径。

---

## 数据准备

1. 克隆仓库后进入 `task1/`。
2. 从网盘解压以下目录到 `task1/` 下（**目录名保持不变**）：

```
task1/
├── submission_assets/     # 输入图像、COLMAP、训练输出、Blender 工程
└── third_party/           # 2d-gaussian-splatting、colmap、Magic123、threestudio
```

`submission_assets/` 主要子目录：

| 目录 | 内容 |
|------|------|
| `object_A/` | 运动鞋 · 36 视图 · COLMAP + 2DGS（10000 iter） |
| `object_B/` | 毛绒小狗 · threestudio SDS（12000 iter） |
| `object_C/` | 木吉他 · Magic123（coarse/fine 各 5000 iter） |
| `garden/` | Mip-NeRF 360 背景 · 2DGS（10000 iter） |
| `scene_compose.blend` | Blender 融合场景 |
| `wandering.mov` | 环视漫游视频 |

---

## 训练（Train）

四条流水线依次运行（**会覆盖** `submission_assets/` 内对应输出）：

```bash
cd task1
conda activate task1

bash scripts/pipeline/object_a.sh          # 物体 A：COLMAP + 2DGS
bash scripts/pipeline/object_b.sh          # 物体 B：threestudio SDS
bash scripts/pipeline/object_c.sh          # 物体 C：Magic123
bash scripts/pipeline/background.sh garden   # 背景：garden + 2DGS
```

各阶段墙钟耗时记录在 `outputs/timing.csv`；训练指标 CSV 在 `outputs/data/`。

---

## 测试 / 查看结果（Test）

题目一无独立评测脚本；解压网盘数据后可直接查看已训练结果，无需重训：

```bash
# 查看各物体 mesh 路径
# 物体 A : submission_assets/object_A/2dgs_output/train/ours_10000/fuse_post.ply
# 物体 B : submission_assets/object_B/save/it12000-export/model.obj
# 物体 C : submission_assets/object_C/magic123_output/object_C_fine/mesh/mesh.obj
# 背景   : submission_assets/garden/2dgs_output/train/ours_10000/fuse_unbounded_post.ply

# 融合场景与视频
open submission_assets/scene_compose.blend   # 需安装 Blender
open submission_assets/wandering.mov
```

重新训练后，2DGS 流水线会自动执行 `render.py` 导出渲染与 mesh；Magic123 / threestudio 的最终 mesh 见上文路径。

---

# 题目二：LeRobot ACT 跨环境泛化

## 环境配置

### 依赖概览

| 项目 | 值 |
|------|-----|
| Conda 环境名（服务器 / 本地冒烟） | `task2` |
| Python | 3.12 |
| PyTorch | 2.10.0 + CUDA 12.1（服务器） |
| LeRobot | 0.5.1 |
| 随机种子 | `SEED=42` |

| 文件 | 用途 |
|------|------|
| [`task2/environment.yml`](task2/environment.yml) | GPU 服务器 conda 环境（同 `environment-server.yml`） |
| [`task2/environment-server.yml`](task2/environment-server.yml) | 同上 |
| [`task2/environment-local.yml`](task2/environment-local.yml) | macOS 本地冒烟 conda 环境 |
| [`task2/requirements-server.txt`](task2/requirements-server.txt) | 服务器 pip 依赖 |
| [`task2/requirements-local.txt`](task2/requirements-local.txt) | 本地 pip 依赖 |

`requirements-server.txt` 核心包：

```
lerobot==0.5.1
av>=15.0.0,<16.0.0
matplotlib>=3.10.0
pyyaml>=6.0
tqdm>=4.66.0
wandb>=0.24.0,<0.25.0
```

### GPU 服务器（conda，推荐）

```bash
cd task2
bash scripts/setup_server.sh
conda activate task2
```

### GPU 服务器（venv 备选）

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
cd task2
bash scripts/setup_local.sh
source .venv/bin/activate
```

验证环境：

```bash
python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__)"
```

> 所有 `task2/scripts/*.sh` 会自动激活 `.venv` 或 conda 环境 `task2`。

---

## 数据准备

### 1. 数据集

将网盘中的五个 LeRobot v3.0 目录解压到 `task2/data/`，**目录名不可改**：

| 目录 | Episodes | 用途 |
|------|----------|------|
| `data/calvin_env_A` | 4468 | ABC 混合训练源 |
| `data/calvin_env_B` | 4468 | B 单环境训练 |
| `data/calvin_env_C` | 4467 | ABC 混合训练源 |
| `data/calvin_env_ABC` | 13403 | ABC_fair 混合训练 |
| `data/calvin_env_D` | 4467 | Zero-shot 评测（训练不可见） |

格式：`meta/info.json` + parquet + AV1 mp4。

校验数据：

```bash
cd task2
python scripts/verify_dataset.py    # 全部 [OK] 后再训练/评测
```

### 2. 模型权重（仅评测时需要）

从网盘解压 `outputs/train_b/` 与 `outputs/train_abc_fair/` 到 `task2/outputs/`。**在保留 GitHub 仓库已有 `outputs/` 评测 JSON 的基础上合并，勿直接覆盖整个 `outputs/` 目录。**

每个 checkpoint 的 `pretrained_model/` 须含 **7 个文件**（`model.safetensors`、`config.json`、`train_config.json`、预处理/后处理 JSON 与 safetensors）。

```bash
cd task2
# 若只有 080000 步目录，为评测脚本创建 last 软链接：
ln -sfn 080000 outputs/train_b/checkpoints/last
ln -sfn 080000 outputs/train_abc_fair/checkpoints/last

# 验证权重就绪
ls outputs/train_b/checkpoints/last/pretrained_model/model.safetensors
ls outputs/train_abc_fair/checkpoints/last/pretrained_model/model.safetensors
```

---

## 训练（Train）

```bash
cd task2
export SEED=42

bash scripts/train.sh B server          # → outputs/train_b/（约 2 小时，80k 步）
bash scripts/train.sh ABC_fair server   # → outputs/train_abc_fair/（ABC 混合，80k 步）
```

- 每 10k 步保存 checkpoint；80k 结束后 `checkpoints/last` 自动更新。
- 若目录已存在需先删除：`rm -rf outputs/train_b`（或 `train_abc_fair`）。
- 本地冒烟（20 步）：`bash scripts/run_smoke.sh`

---

## 测试（Test）

```bash
cd task2
export SEED=42

# 完整评测 + 报告 JSON/图表
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair

# 仅 Zero-shot → D 环境
bash scripts/eval_zeroshot.sh B server
bash scripts/eval_zeroshot.sh ABC_fair server

# 快速自检（50 batch）
bash scripts/diagnose.sh B
bash scripts/diagnose.sh ABC_fair
```

主要输出：

- `outputs/eval_*_on_*.json` — 各环境离线评测
- `outputs/report_*.json` — 汇总报告
- `outputs/gap_comparison.png`、`outputs/chunk_l1_*.png` — 对比图表

---

## 常见问题

### 题目一

| 问题 | 处理 |
|------|------|
| `third_party/` 缺失 | 从网盘解压到 `task1/third_party/` |
| COLMAP / 2DGS 找不到 | 确认已 `conda activate task1` 且路径在 `scripts/common.sh` 中正确 |
| 仅查看融合结果 | 解压网盘后打开 `submission_assets/scene_compose.blend` 与 `wandering.mov` |

### 题目二

| 问题 | 处理 |
|------|------|
| `No checkpoint under .../checkpoints/last` | `ln -sfn 080000 outputs/train_*/checkpoints/last` |
| `Output exists: outputs/train_b` | 重新训练前 `rm -rf outputs/train_b` |
| `ModuleNotFoundError: lerobot` | 重新执行环境配置并 `conda activate task2` 或 `source .venv/bin/activate` |
| 数据集校验失败 | 检查 `data/calvin_env_*` 目录名与 `meta/info.json` 是否完整 |

---

## 快速复现清单

```bash
# ===== 题目一 =====
cd task1
bash scripts/setup_server.sh && conda activate task1
# 解压网盘 submission_assets/ 与 third_party/ 到 task1/
bash scripts/pipeline/object_a.sh
bash scripts/pipeline/object_b.sh
bash scripts/pipeline/object_c.sh
bash scripts/pipeline/background.sh garden

# ===== 题目二 =====
cd task2
bash scripts/setup_server.sh && conda activate task2
# 解压网盘 calvin_env_* 到 task2/data/，权重到 task2/outputs/
export SEED=42
bash scripts/eval_report.sh B server
bash scripts/eval_report.sh ABC_fair server
bash scripts/eval_compare.sh B ABC_fair
# 如需重训：
# bash scripts/train.sh B server
# bash scripts/train.sh ABC_fair server
```
