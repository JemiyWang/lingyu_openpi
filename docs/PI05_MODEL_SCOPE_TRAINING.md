# 在新电脑上使用 ModelScope 数据集微调官方 π0.5

本文档说明如何在另一台 Linux/NVIDIA GPU 电脑上：

1. 下载定制版 OpenPI 代码；
2. 下载官方 `pi05_base` 权重；
3. 从 ModelScope 下载 `jemiywang/sec_robot_lingyu` 数据集；
4. 为六个任务分别准备 SED TeleAvatar 数据配置；
5. 使用各任务数据集内已有的 `norm_stats.json`，并分别从官方 `pi05_base` 开始 fine-tuning。
6. 在远端训练机器登录 W&B，并将训练日志上传到指定账号或团队。

代码仓库：<https://github.com/JemiyWang/lingyu_openpi>

ModelScope 数据集：<https://www.modelscope.cn/datasets/jemiywang/sec_robot_lingyu>

官方 OpenPI：<https://github.com/Physical-Intelligence/openpi>

## 0. 先确认硬件和磁盘

本数据集目前约 115 GiB，下载后还需要保存模型缓存、训练 checkpoint 和日志，建议训练盘至少有 300 GiB 可用空间。

本文只使用完整 fine-tuning，不使用 LoRA。官方 OpenPI 给出的完整 fine-tuning 显存参考是大于 70 GiB，建议使用 A100 80 GB 或 H100。当前训练脚本支持单机多卡，但不支持多机训练。

参考官方 [OpenPI README](https://github.com/Physical-Intelligence/openpi#requirements)。

本文默认使用 JAX 进行完整 fine-tuning。如果新电脑显存不足以进行完整 fine-tuning，应更换训练机器或增加显存，不切换到 LoRA。

## 1. 安装系统依赖

建议使用 Ubuntu 22.04、Python 3.11、NVIDIA 驱动和可用的 CUDA GPU。确认 GPU 正常：

```bash
nvidia-smi
```

安装 Git、FFmpeg 和 `uv`：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

重新打开一个 shell，或按 `uv` 安装提示更新 `PATH`。

## 2. 下载代码并安装 OpenPI 环境

将代码放在大容量磁盘，例如 `/data`：

```bash
mkdir -p /data
cd /data

git clone --recurse-submodules \
  https://github.com/JemiyWang/lingyu_openpi.git \
  openpi

cd /data/openpi

uv python install 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` 是官方 OpenPI 安装说明中的必要设置，用于避免拉取 LeRobot 依赖时自动下载不需要的 LFS 文件。

检查代码版本和环境：

```bash
git remote -v
git log -1 --oneline
uv run python -c "import jax, torch; print('jax', jax.__version__); print('torch', torch.__version__)"
```

## 3. 设置 OpenPI 缓存目录

不要把模型缓存放在容量较小的系统盘。官方 OpenPI 默认使用 `~/.cache/openpi`，这里改到数据盘：

```bash
mkdir -p /data/cache/openpi
export OPENPI_DATA_HOME=/data/cache/openpi
```

如果希望每次登录后自动生效，可以将上面的 `export` 写入当前用户的 shell 配置文件。

## 4. 下载官方 π0.5 Base 权重

官方用于 fine-tuning 的 π0.5 基础权重路径是：

```text
gs://openpi-assets/checkpoints/pi05_base
```

预先下载整个 checkpoint 和相关资产：

```bash
cd /data/openpi

OPENPI_DATA_HOME=/data/cache/openpi \
uv run python -c \
"from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
```

训练配置中应使用：

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
)
```

训练时 OpenPI 也会自动下载这个路径；提前下载的好处是可以先验证新电脑是否能够访问 Google Cloud Storage，并避免训练开始后才发现网络或磁盘问题。

## 5. 安装 ModelScope Hub 并下载数据集

安装 ModelScope Hub CLI：

```bash
python3 -m pip install -U modelscope-hub
```

登录 ModelScope。Token 只在终端的隐藏输入框中输入，不要写入脚本或聊天记录：

```bash
ms-hub login
ms-hub whoami
```

下载数据集到本地：

```bash
mkdir -p /data/datasets

ms-hub download jemiywang/sec_robot_lingyu \
  --repo-type dataset \
  --local-dir /data/datasets/sec_robot_lingyu \
  --max-workers 8
```

检查下载结果：

```bash
find /data/datasets/sec_robot_lingyu \
  -path '*/meta/info.json' \
  -print
```

每个 LeRobot 数据集都应该有自己的 `meta/info.json`、`meta/tasks.jsonl`、`data/` 和 `videos/`。

## 6. 配置 Weights & Biases（W&B）

OpenPI 训练脚本已经启用 W&B：训练 loss、训练步数以及首批相机图像会在线记录到 W&B。这里必须在**远端训练机器**上登录你要使用的 W&B 账号；不要把 API Key 写入代码、shell 脚本、Git 仓库或聊天记录。

先在远端机器执行：

```bash
cd /data/openpi

# 在隐藏输入框中粘贴 W&B API Key
uv run wandb login --relogin
#直接读取key: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
# 个人账号名或团队名，例如 your_wandb_username / your_wandb_team
export WANDB_ENTITY="your_wandb_username_or_team"

# 在线上传训练日志；不要设置为 offline
export WANDB_MODE=online
```

`WANDB_ENTITY` 决定 run 上传到哪个 W&B 用户或团队。当前项目的默认 W&B project 是 `openpi`；如果要使用其他 project，在每次训练命令中加入：

```bash
--project-name=your_wandb_project
```

例如：

```bash
OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --project-name=pi05_teleavatar \
  --exp-name=pi05_teleavatar_collect_food \
  --overwrite
```

如果登录的 API Key 属于个人账号，`WANDB_ENTITY` 填个人用户名；如果要上传到团队 project，API Key 对应的账号必须有该团队的写权限。W&B 只记录训练指标和图像，checkpoint 仍保存在训练机器本地，不会自动作为 W&B Artifact 上传。

## 7. 重要：确认六个数据集的目录结构

标准 OpenPI/LeRobot 数据加载器一次读取一个数据集根目录，要求该目录直接包含：

```text
<dataset_root>/meta/info.json
<dataset_root>/data/...
<dataset_root>/videos/...
```

已检查 ModelScope 远端 `jemiywang/sec_robot_lingyu` 的实际文件树。仓库根目录包含说明文件和六个任务目录，六个任务目录各自都是独立的 LeRobot 数据集：

```text
sec_robot_lingyu/
├── .gitattributes
├── README.md
├── dataset_infos.json
├── collect_food/
├── fold_towels/
├── pick_flowers/
├── pick_up_paper_rolls/
├── pick_up_trash/
└── stack_blocks/
```

每个任务目录直接包含自己的 `meta/info.json`、`meta/tasks.jsonl`、`data/`、`videos/` 和 `norm_stats.json`。因此训练时不要把 `/data/datasets/sec_robot_lingyu` 这个父目录作为 `repo_id`，而要指向对应的任务子目录。本文采用“六个任务分别训练六个模型”的方式，不合并数据集，也不在任务之间续训。

每次训练的 `repo_id` 必须指向当前任务自己的 LeRobot 根目录：

| 任务 | `repo_id` 示例 |
|---|---|
| `collect_food` | `/data/datasets/sec_robot_lingyu/collect_food` |
| `fold_towels` | `/data/datasets/sec_robot_lingyu/fold_towels` |
| `pick_flowers` | `/data/datasets/sec_robot_lingyu/pick_flowers` |
| `pick_up_paper_rolls` | `/data/datasets/sec_robot_lingyu/pick_up_paper_rolls` |
| `pick_up_trash` | `/data/datasets/sec_robot_lingyu/pick_up_trash` |
| `stack_blocks` | `/data/datasets/sec_robot_lingyu/stack_blocks` |

如果某个任务目录下没有 `meta/info.json`，不要开始训练；先确认数据是否完整下载。

建议先完成一个任务的 smoke test，再按相同流程依次训练剩余五个任务。六个任务不需要合并；每次只修改 `pi05_teleavatar` 配置中的当前任务路径，然后从 `pi05_base` 重新开始训练。

## 8. SED 数据格式和代码检查

该仓库中包含针对 SED 数据集新增的定制适配代码（不是官方 OpenPI 原生组件）：

- `src/openpi/sed_robot_adapter.py`
- `src/openpi/sed_robot_data_loader.py`

这两个文件只有在接入 `pi05_teleavatar` 的正式训练路径后才会生效。文件存在本身并不代表标准训练命令会自动调用它们：

- `sed_robot_adapter.py` 负责把 SED 的相机、state 和 action 字段转换成 PI05 所需的输入格式；
- `sed_robot_data_loader.py` 负责 SED 视频读取、时间戳容差和解码性能处理；
- 正式训练必须让 `name="pi05_teleavatar"` 使用 `LeRobotTeleavatarDataConfig(use_sed_schema=True)`，由该配置接入上述 SED 适配器和数据加载器；
- 不得改用 `pi05_sed_robot`、`train_sed_robot_lora.py` 或任何 LoRA 配置来绕过这条路径。

当前 SED 数据格式是：

```text
相机：top_head、hand_left、hand_right
state：16 维
action：16 维
动作：绝对关节目标，默认不转换为 delta action
```

适配器会将相机映射为：

```text
top_head  -> base_0_rgb
hand_left -> left_wrist_0_rgb
hand_right -> right_wrist_0_rgb
```

训练前检查 SED 适配代码是否已经被训练数据加载流程调用：

```bash
cd /data/openpi
rg -n "pi05_teleavatar|sed_robot_adapter|sed_robot_data_loader|create_torch_dataset" \
  src/openpi
```

检查 `pi05_teleavatar` 的配置时，必须确认它仍然使用配置名 `pi05_teleavatar`，并且其 `LeRobotTeleavatarDataConfig` 已开启 `use_sed_schema=True`：

```bash
rg -n -A25 'name="pi05_teleavatar"' src/openpi/training/config.py
rg -n 'use_sed_schema=True|use_sed_video_loader|sed_robot_adapter|sed_robot_data_loader' \
  src/openpi scripts
```

如果 `pi05_teleavatar` 没有显示 `use_sed_schema=True`，说明 SED 适配尚未接入该配置，不要直接开始正式训练。

注意：六次训练都必须使用仓库中已有的 `pi05_teleavatar` 配置名。修改数据路径后，还要确认 `sed_robot_adapter.py` 和 `sed_robot_data_loader.py` 已经接入该配置的训练路径，并且 SED 数据字段已经映射到 PI05 的标准输入字段。不要为了不同任务新建配置名称。

六个任务统一使用项目配置文件中已有的：

```python
name="pi05_teleavatar"
```

不新增其他训练配置，也不使用 `pi05_droid` 或任何 LoRA 配置。每次训练前，只修改 `pi05_teleavatar` 中当前任务对应的 `repo_id`、`assets_dir` 和 `asset_id`。

以 `collect_food` 为例：

```python
TrainConfig(
    name="pi05_teleavatar",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=30,
    ),

    data=LeRobotTeleavatarDataConfig(
        repo_id="/data/datasets/sec_robot_lingyu/collect_food",
        assets=AssetsConfig(
            # assets_dir 是数据集父目录，asset_id 指向当前任务目录
            assets_dir="/data/datasets",
            asset_id="sec_robot_lingyu/collect_food",
        ),
        default_prompt="Collect the food.",
        use_sed_schema=True,
    ),

    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
)
```

`action_horizon=30` 是当前项目 Teleavatar 配置使用的示例值，实际值应根据机器人控制频率和希望输出的动作 chunk 长度确认，不应盲目照搬。

每个任务都必须保留上面的 `pi05_base` 权重加载器，不要把前一个任务的训练 checkpoint 填到下一个任务；六个训练是六次独立初始化。

同时确认 `LeRobotTeleavatarDataConfig(use_sed_schema=True)` 已经通过仓库中的适配代码把 SED 数据字段映射到 PI05 的标准输入字段，而不是使用旧 TeleAvatar 字段映射。

## 9. 使用数据集内已有的 normalization statistics

六个任务的 `norm_stats.json` 已经随数据集上传，因此本文流程不再运行 `compute_norm_stats.py`。

以 `collect_food` 为例，文件应位于：

```text
/data/datasets/sec_robot_lingyu/collect_food/norm_stats.json
```

因此 `pi05_teleavatar` 中的配置应让 OpenPI 从数据集目录加载该文件：

```python
assets=AssetsConfig(
    assets_dir="/data/datasets",
    asset_id="sec_robot_lingyu/collect_food",
)
```

这会读取：

```text
/data/datasets/sec_robot_lingyu/collect_food/norm_stats.json
```

换任务时，只替换 `repo_id`、`assets_dir` 和 `asset_id`，例如：

```text
collect_food       -> sec_robot_lingyu/collect_food
fold_towels        -> sec_robot_lingyu/fold_towels
pick_flowers       -> sec_robot_lingyu/pick_flowers
pick_up_paper_rolls -> sec_robot_lingyu/pick_up_paper_rolls
pick_up_trash      -> sec_robot_lingyu/pick_up_trash
stack_blocks       -> sec_robot_lingyu/stack_blocks
```

六个任务的 `asset_id` 都指向对应的任务子目录；`pick_up_paper_rolls` 使用 `sec_robot_lingyu/pick_up_paper_rolls`，不使用仓库根目录。

训练前检查当前任务的统计文件：

```bash
test -f /data/datasets/sec_robot_lingyu/collect_food/norm_stats.json \
  && echo "collect_food norm stats OK"
```

如果出现 `Normalization stats not found`，优先检查 `assets_dir`、`asset_id` 和当前任务目录，不要重新计算或混用其他任务的统计文件。

## 10. 分别启动六次 π0.5 fine-tuning

以 `collect_food` 为例：

```bash
cd /data/openpi

OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py \
  pi05_teleavatar \
  --exp-name=pi05_teleavatar_collect_food \
  --overwrite
```

其他五个任务分别执行对应配置：

```bash
OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --exp-name=pi05_teleavatar_fold_towels --overwrite

OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --exp-name=pi05_teleavatar_pick_flowers --overwrite

OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --exp-name=pi05_teleavatar_pick_up_paper_rolls --overwrite

OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --exp-name=pi05_teleavatar_pick_up_trash --overwrite

OPENPI_DATA_HOME=/data/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_teleavatar \
  --exp-name=pi05_teleavatar_stack_blocks --overwrite
```

以上六条命令是六次独立训练。每次训练都从：

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

加载初始权重，不从前一个任务的 checkpoint 继续训练，也不使用前一个任务的 `resume`。

训练日志和 checkpoint 默认分别保存在：

```text
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_collect_food/
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_fold_towels/
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_pick_flowers/
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_pick_up_paper_rolls/
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_pick_up_trash/
/data/openpi/checkpoints/pi05_teleavatar/pi05_teleavatar_stack_blocks/
```

官方 OpenPI 的通用示例通常会先运行 `compute_norm_stats.py`，但本项目的六个数据集已经包含各自的 `norm_stats.json`，所以这里直接使用数据集内已有统计量；`--overwrite` 会覆盖同名实验目录下的已有结果。训练流程参考官方 [Fine-Tuning Base Models on Your Own Data](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data)。

## 11. 训练过程中的常见问题

### 显存不足

本文不使用 LoRA。显存不足时只能更换显存更大的训练机器、增加单机 GPU 数量或使用 FSDP；不要改用 LoRA 配置。

### 视频时间戳不匹配

SED 视频是 HEVC，且视频时间戳和 parquet 时间戳可能存在小偏差。应使用仓库中的 `sed_robot_data_loader.py`，不要简单替换成标准 `LeRobotDataset` 解码路径。

### Action dimension mismatch

确认：

- 原始数据的 `action` 是 16 维；
- π0.5 模型配置的 `action_dim` 是 32；
- SED 适配器负责将模型输出还原为前 16 维物理动作。

### 训练读不到数据

检查 `repo_id` 是否指向真正包含 `meta/info.json` 的目录，而不是只包含六个子目录的父目录。

### 系统盘被占满

检查以下目录是否位于大容量数据盘：

```bash
echo "$OPENPI_DATA_HOME"
du -sh /data/cache/openpi /data/datasets/sec_robot_lingyu /data/openpi/checkpoints
```

## 最终执行顺序

```text
下载 JemiyWang/lingyu_openpi
        ↓
安装 uv 环境和 OpenPI 依赖
        ↓
设置 OPENPI_DATA_HOME 到大容量磁盘
        ↓
下载官方 gs://openpi-assets/checkpoints/pi05_base
        ↓
登录 ModelScope
        ↓
下载 jemiywang/sec_robot_lingyu
        ↓
登录 W&B 并设置 `WANDB_ENTITY`、`WANDB_MODE=online`
        ↓
检查六个 LeRobot 数据集的目录结构
        ↓
复用 `name="pi05_teleavatar"`，切换当前任务的 repo_id 和 assets
        ↓
确认 SED loader/config 已接入
        ↓
确认每个任务数据集内已有的 norm_stats.json
        ↓
分别启动六次 pi05 fine-tuning
        ↓
得到六个独立的训练 checkpoint
```

策略服务启动、机器人推理和评估属于部署流程，不在本文的训练流程范围内。
