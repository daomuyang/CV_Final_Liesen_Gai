# 题目一：多物体 3D 重建与场景融合

基于 COLMAP、2D Gaussian Splatting、threestudio（SDS）与 Magic123，完成三个物体与 Mip-NeRF 360 背景的重建，并在 Blender 中融合为最终场景。

## 提交分工

| 渠道 | 内容 |
|------|------|
| **GitHub** | 代码（`scripts/`、`patches/`）、`environment.yml`、`outputs/timing.csv`、`outputs/data/` |
| **百度网盘** | `submission_assets/`（全部输入 + 输出数据，约 8.8 GB）、`third_party/`（全部第三方库，约 200 MB） |

网盘链接：https://pan.baidu.com/s/1dlkCUwM8b2_L2cgbFi3RKA?pwd=stq2  
提取码：`stq2`

流水线脚本通过 `DATASET_ROOT=submission_assets/` 读写数据，通过 `THIRD_PARTY_DIR=third_party/` 调用第三方库（见 `scripts/common.sh`）。

---

## 仓库结构（GitHub）

以下为 GitHub 仓库中的 `task1/` 目录结构（不含网盘提供的 `submission_assets/` 与 `third_party/`）：

```
task1/
├── README.md
├── environment.yml              # Conda 环境（cvpj1）
├── patches/                     # 对第三方库的补丁
│   ├── 2d-gaussian-splatting-alpha-mask.patch
│   └── Magic123-wandb.patch
├── scripts/
│   ├── common.sh                # 路径常量、公共函数
│   ├── setup_server.sh          # 服务器环境安装
│   ├── timing.sh                # 墙钟计时
│   ├── wandb.sh                 # WandB 离线日志
│   ├── pipeline/                # 四条流水线入口
│   │   ├── object_a.sh          # 物体 A：COLMAP + 2DGS
│   │   ├── object_b.sh          # 物体 B：threestudio SDS
│   │   ├── object_c.sh          # 物体 C：Magic123
│   │   └── background.sh        # 背景：garden + 2DGS
│   └── tools/
│       ├── preprocess_image.py
│       ├── plot_magic123_loss.py
│       ├── log_metrics_to_wandb.py
│       ├── blender_vertex_color.py
│       └── run_blender_color.sh
└── outputs/
    ├── timing.csv               # 各阶段墙钟耗时
    └── data/                    # 训练指标 CSV（供报告/WandB）
        ├── twodgs:*.csv
        ├── dreamfusion:loss_*.csv
        └── magic123:loss_*.csv
```

网盘中的 `third_party/` 包含：`2d-gaussian-splatting`、`colmap`、`Magic123`、`threestudio`。

---

## 网盘数据目录（`submission_assets/`）

```
submission_assets/
├── object_A/                         # 运动鞋 · COLMAP + 2DGS（10000 iter）
│   ├── images_raw/                   # 36 张自采原图
│   ├── images/                       # 抠图前景
│   ├── sparse/                       # COLMAP 稀疏重建
│   ├── database.db
│   └── 2dgs_output/
│       ├── train/ours_10000/         # fuse_post.ply、vis/、renders/、gt/
│       └── point_cloud/iteration_10000/
├── object_B/                         # 毛绒小狗 · threestudio SDS（12000 iter）
│   ├── ckpts/last.ckpt
│   ├── configs/parsed.yaml, raw.yaml
│   ├── csv_logs/
│   ├── cmd.txt
│   ├── tb_logs/
│   └── save/
│       ├── it12000-export/           # model.obj + model.mtl + texture_kd.jpg
│       ├── it12000-0.png, it12000-test.mp4
│       └── it{step}-0.png           # 训练过程预览
├── object_C/                         # 木吉他 · Magic123（coarse/fine 各 5000 iter）
│   ├── images/                       # 0001.jpg、rgba.png 等
│   └── magic123_output/
│       ├── object_C_coarse/          # mesh/、checkpoints/、run/、training/、validation/、results/
│       └── object_C_fine/
├── garden/                           # Mip-NeRF 360 背景 · 2DGS（10000 iter）
│   ├── images/、images_2/、images_4/、images_8/
│   ├── sparse/、poses_bounds.npy
│   └── 2dgs_output/
│       ├── train/ours_10000/         # fuse_unbounded_post.ply、vis/、renders/、gt/
│       └── point_cloud/iteration_10000/
├── scene_compose.blend               # Blender 融合工程
└── wandering.mov                     # 环视漫游视频
```

---

## 复现

```bash
# 1. 克隆 GitHub 仓库
cd task1
bash scripts/setup_server.sh
conda activate cvpj1

# 2. 下载百度网盘，解压到 task1/ 下：
#    submission_assets/  → task1/submission_assets/
#    third_party/        → task1/third_party/

# 3. 查看结果（无需重训）
#    物体 A mesh : submission_assets/object_A/2dgs_output/train/ours_10000/fuse_post.ply
#    物体 B mesh : submission_assets/object_B/save/it12000-export/model.obj
#    物体 C mesh : submission_assets/object_C/magic123_output/object_C_fine/mesh/mesh.obj
#    背景 mesh   : submission_assets/garden/2dgs_output/train/ours_10000/fuse_unbounded_post.ply
#    融合场景    : submission_assets/scene_compose.blend
#    视频        : submission_assets/wandering.mov

# 4. 重新训练（会覆盖 submission_assets 内对应输出）
bash scripts/pipeline/object_a.sh
bash scripts/pipeline/object_b.sh
bash scripts/pipeline/object_c.sh
bash scripts/pipeline/background.sh garden
```

**仅查看融合场景**：解压网盘后打开 `scene_compose.blend` 与 `wandering.mov` 即可（仍需 `third_party/` 若需重训）。

---

## 各流水线说明

### 物体 A（COLMAP + 2DGS，10000 iter）

| 阶段 | 输出路径 |
|------|----------|
| preprocess | `object_A/images/` |
| colmap | `object_A/sparse/` |
| 2dgs_train + export | `object_A/2dgs_output/train/ours_10000/`、`point_cloud/iteration_10000/` |

### 物体 B（threestudio SDS，12000 iter）

权重：`object_B/ckpts/last.ckpt`；导出 mesh：`object_B/save/it12000-export/`。

### 物体 C（Magic123，5000 + 5000 iter）

输入：`object_C/images/`；最终 mesh：`object_C/magic123_output/object_C_fine/mesh/`。

### 背景（garden + 2DGS，10000 iter）

```bash
bash scripts/pipeline/background.sh garden
```

读取 `submission_assets/garden/`（`sparse/` + `images/`）；mesh 输出：`garden/2dgs_output/train/ours_10000/fuse_unbounded_post.ply`。

---

## 提交清单

| 项目 | 位置 |
|------|------|
| GitHub | `task1/` 代码 + `outputs/` |
| 百度网盘 | `submission_assets/` + `third_party/` |
| 报告 PDF | 含网络结构、超参、Loss、WandB 曲线 |

---

## 作者

盖烈森 · 23307130013@m.fudan.edu.cn
