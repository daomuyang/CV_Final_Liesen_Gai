# CALVIN 仿真环境配置与评测指南

> 适用于在**离线训练已完成**后，用 CALVIN 官方仿真器测量真实 **Success Rate**。

## 重要说明

| 项目 | 说明 |
|------|------|
| 当前 LeRobot 训练 | **不需要**仿真环境，用 `data/calvin_env_*` 即可 |
| 官方 `evaluate_policy.py` | 仅支持 **CALVIN baseline** 模型格式，**不能**直接加载 LeRobot ACT checkpoint |
| 本仓库离线评测 | `bash scripts/eval_zeroshot.sh ...`（Action L1 + 代理 success_rate） |
| 仿真 Success Rate | 需单独安装 CALVIN，并编写 **LeRobot → CALVIN env** 的推理桥接 |

因此：**装仿真环境不会改变你已训好的模型**；只是多一种评测方式。

---

## 一、安装 CALVIN 仿真（推荐独立 conda 环境）

CALVIN 官方要求 **Python 3.8**，与 task2 的 Python 3.12/3.13 **分开**，避免依赖冲突。

```bash
# 1. 克隆（与 task2 并列目录即可）
cd /mnt/workspace   # 或你的服务器工作目录
git clone https://github.com/mees/calvin.git
cd calvin

# 2. 创建环境
conda create -n calvin_py38 python=3.8 -y
conda activate calvin_py38

# 3. 安装（若 pyhash 失败：pip install 'setuptools<58' 后重试）
sh install.sh

# 4. 验证仿真能否 import
python -c "from calvin_env.envs.play_table_env import PlayTableSimEnv; print('OK')"
```

### 下载 CALVIN 官方评测用数据（仿真 rollout 需要）

离线 LeRobot 数据（`calvin_env_D`）与官方仿真评测数据**不是同一套路径格式**。要做标准 CALVIN Challenge 评测，需下载官方 split：

```bash
cd calvin/dataset
sh download_data.sh D        # 环境 D，约 166 GB
# 或调试：sh download_data.sh debug   # 约 1.3 GB
```

语言 embedding（评测条件策略时需要）：

```bash
sh download_lang_embeddings.sh D
```

---

## 二、官方评测命令（仅适用于 CALVIN baseline）

```bash
cd calvin/calvin_models/calvin_agent

python evaluation/evaluate_policy.py \
  --dataset_path /path/to/calvin/dataset/task_D_D \
  --train_folder /path/to/calvin_baseline_train_log
```

- 默认加载 `train_folder` 下**最后一个** checkpoint
- `--checkpoint /path/to/ckpt` 可指定权重
- `--debug` 可可视化

**局限**：该脚本期望 CALVIN 自研 policy（MCIL 等），**无法直接读取** `outputs/train_a/checkpoints/last/pretrained_model/` 下的 LeRobot ACT。

---

## 三、用 LeRobot ACT 在 CALVIN 仿真里评测（需自行桥接）

思路：在 CALVIN `PlayTableSimEnv` 里逐步 rollout，每步把观测转成 LeRobot 格式，调用已训 ACT，再把动作送回环境。

### 观测 / 动作对齐要点

| LeRobot（本作业数据） | CALVIN 仿真 |
|----------------------|-------------|
| `observation.images.image` 256×256 | static RGB（需 resize） |
| `observation.images.wrist_image` 256×256 | gripper RGB |
| `observation.state` 15 维 | robot proprio（需对齐字段顺序） |
| `action` 7 维 | relative/absolute tcp（见 CALVIN 文档） |

### 桥接脚本伪代码流程

```python
# 在 task2 的 .venv 中（有 lerobot）
from pathlib import Path
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

ckpt = Path("outputs/train_a/checkpoints/last/pretrained_model")
cfg = PreTrainedConfig.from_pretrained(ckpt)
policy = get_policy_class(cfg.type).from_pretrained(ckpt).cuda().eval()
preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(ckpt))

# 在 calvin_py38 环境中启动 PlayTableSimEnv，或通过 subprocess/IPC 传观测
# 每步：
#   obs_sim -> dict(image, wrist_image, state) -> preprocessor -> policy -> postprocessor -> action_sim
#   env.step(action_sim)
# 按 CALVIN 协议统计 multi-step task success（MTLC / LH-MTLC）
```

完整桥接需处理：动作空间（relative vs absolute）、episode 初始状态、语言条件（若作业要求）、Success 判定逻辑（CALVIN `evaluate_policy.py` 内有参考实现）。

**实用建议（课程作业）**：

1. **报告主结果**：继续用 `eval_zeroshot.sh` 的离线指标 + WandB 训练曲线（与多数同学一致）。
2. **加分 / 探究**：装好 CALVIN 后，用 `debug` 模式目视检查 policy 在 D 仿真里的行为是否合理。
3. 若需完整 MTLC Success Rate，需投入额外开发桥接脚本（非本仓库现成功能）。

---

## 四、推荐工作流（你当前阶段）

```bash
# === task2 环境（已有）===
cd /mnt/workspace/cv_final/task2
source .venv/bin/activate

# 1. 主实验（若尚未完成）
bash scripts/train.sh B server
bash scripts/train.sh ABC_fair server          # ~2h
bash scripts/eval_zeroshot.sh B server
bash scripts/eval_zeroshot.sh ABC_fair server

# 2. 对比离线结果
#    outputs/report_B.json
#    outputs/report_ABC_FAIR.json

# === 可选：CALVIN 仿真（独立 conda）===
conda activate calvin_py38
cd /mnt/workspace/calvin/dataset && sh download_data.sh debug
# 验证环境后，再考虑桥接 LeRobot policy
```

---

## 五、常见问题

**Q：仿真环境和 task2 能共用一个 venv 吗？**  
A：不建议。CALVIN 锁定 Python 3.8 + 旧版 PyTorch；task2 用 3.12 + lerobot 0.5.1。

**Q：装了仿真，要重新训练吗？**  
A：不需要。权重不变，只是评测方式不同。

**Q：离线 1% success_rate 和仿真 Success Rate 差很多？**  
A：正常。离线阈值很严，且不是任务级成功；仿真按整条语言任务链统计。

更多细节见 [CALVIN 官方仓库](https://github.com/mees/calvin) 与 [Leaderboard](http://calvin.cs.uni-freiburg.de/)。
