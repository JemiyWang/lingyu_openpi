# OpenPI 项目修改文档

本文档记录了相对于原始 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 项目所做的所有修改。

## 概述

本项目在原始 OpenPI 基础上增加了 **Teleavatar 双臂机器人** 的完整支持，包括：
- 双臂7自由度+夹爪的控制流程
- 创新的 Soft Inpainting 动作约束方法
- 关节空间和末端执行器两种控制表示
- 完整的 ROS2 集成和部署流程

---

## 1. 新增策略文件

### 1.1 关节空间策略

**文件**: `src/openpi/policies/teleavatar_policy.py`

实现 `TeleavatarInputs` 和 `TeleavatarOutputs` 类，用于 Teleavatar 双臂机器人的关节空间控制：

| 参数 | 维度 | 说明 |
|------|------|------|
| 状态输入 | 48维 | 关节位置、速度、力矩 |
| 模型输入 | 14维 | 左臂7关节 + 右臂7关节 |
| 动作输出 | 16维 | 左臂7关节+1夹爪 + 右臂7关节+1夹爪 |
| 摄像头 | 3个 | left_color, right_color, head_camera |

### 1.2 末端执行器策略

**文件**: `src/openpi/policies/teleavatar_policy_endeffector.py`

末端执行器表示变体，支持笛卡尔空间控制：

| 参数 | 维度 | 说明 |
|------|------|------|
| 状态输入 | 62维 | 扩展状态（含末端位姿）|
| 状态格式 | 16维 | [left_ee_pose(7), left_gripper(1), right_ee_pose(7), right_gripper(1)] |
| 动作类型 | - | Delta 末端位姿 |

---

## 2. 核心模型增强 - Soft Inpainting

### 2.1 配置参数

**文件**: `src/openpi/models/pi0_config.py`

新增配置参数：

```python
use_soft_inpainting: bool = False      # 启用 soft inpainting
time_threshold_inpaint: float = 0.3    # 约束阈值（t > 0.3 时应用约束）
use_correlated_noise: bool = False     # 使用相关噪声
correlation_beta: float = 0.5          # 相关矩阵收缩参数
```

### 2.2 模型实现

**文件**: `src/openpi/models/pi0.py`

代码量从 279 行增加到 486 行（+74%）。

**新增方法**：

| 方法 | 功能 |
|------|------|
| `load_correlation_matrix()` | 加载并应用 beta 收缩的相关矩阵 |
| `_precompute_correction_matrix()` | 预计算相关感知的 inpainting 校正矩阵 |

**增强的 `sample()` 方法**：
- 支持 `initial_actions` 参数指定约束
- 计算 O_indices（inpainted 维度）和 U_indices（自由维度）
- 实现基于相关矩阵的校正传播
- 缓存校正矩阵提升效率

**算法特点**：
- 在 inpainted 维度 (O) 上施加硬约束
- 在自由维度 (U) 上通过相关校正施加软约束
- 使用 epsilon 正则化保证矩阵求逆稳定性

---

## 3. 新增训练配置

**文件**: `src/openpi/training/config.py`

### 3.1 数据配置类

- `LeRobotTeleavatarDataConfig` - 48维关节空间配置
- `LeRobotTeleavatarEndEffectorDataConfig` - 62维末端执行器配置

### 3.2 训练配置

| 配置名 | 基础模型 | 特点 |
|--------|----------|------|
| `pi05_teleavatar` | Pi0.5 | 标准微调 |
| `pi0_teleavatar` | Pi0 | 启用 soft inpainting |
| `pi0_teleavatar_lora` | Pi0 | LoRA 微调（16GB 显存）|
| `pi0_teleavatar_endeffector` | Pi0 | 末端执行器表示 |
| `pi0_teleavatar_low_mem_finetune` | Pi0 | 低显存 LoRA |
| `pi0_teleavatar_low_mem_finetune_endeffector` | Pi0 | 低显存末端执行器 LoRA |

### 3.3 其他配置修改

- 默认 `num_workers` 从 2 增加到 32
- DROID 配置：`datasets` 参数改为 `filter_dict_path`
- 删除未使用的 `polaris_config.py` 导入

---

## 4. 新增 Teleavatar 示例

**目录**: `examples/teleavatar/`

### 4.1 核心脚本

| 文件 | 功能 |
|------|------|
| `main.py` | 主执行脚本（关节空间）|
| `main_endeffector.py` | 主执行脚本（末端执行器）|
| `main_dataset.py` | 基于数据集的执行脚本 |

### 4.2 环境封装

| 文件 | 功能 |
|------|------|
| `env.py` | 基础环境封装 |
| `env_endeffector.py` | 末端执行器环境 |
| `env_dataset.py` | 数据集环境 |
| `dataset_interface.py` | 抽象数据集接口 |

### 4.3 ROS2 集成

| 文件 | 功能 |
|------|------|
| `ros2_interface.py` | ROS2 关节控制接口 |
| `ros2_interface_endeffector.py` | ROS2 末端执行器控制接口 |
| `deploy_policy_bridge.py` | 策略部署桥接 |

### 4.4 工具脚本

| 文件 | 功能 |
|------|------|
| `convert_teleavatar_data_to_lerobot.py` | 数据格式转换 |
| `arm_pd_controller.py` | PD 控制器 |

### 4.5 文档

- `README_ENDEFFECTOR.md` - 末端执行器表示使用说明

---

## 5. 新增工具脚本

### 5.1 训练脚本

**文件**: `scripts/train_pick_up_paper.sh`

```bash
# 8 GPU FSDP 分布式训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar \
    --exp-name=pick_up_paper \
    --overwrite \
    --batch-size=128 \
    --num-train-steps=50000 \
    --fsdp-devices=8
```

### 5.2 数据集合并工具

**文件**: `scripts/merge_lerobot_datasets.py`

- 合并多个 LeRobot v2.1 数据集
- 处理跨数据集的任务重映射
- 创建统一的 data/videos/meta 目录结构
- 支持可配置的 chunk 大小

### 5.3 相关矩阵计算

**文件**: `scripts/compute_norm_stats.py`

新增 `compute_correlation_matrix()` 函数：
- 计算动作 chunk 的 Cholesky 分解
- 处理常数维度和数值稳定性
- 为非正定矩阵实现正则化

### 5.4 模型下载器

**文件**: `download_pi0.py`

- 从 GCS 下载 pi0_base 检查点
- 支持 gcloud、gsutil 和 Python 下载器
- 默认缓存路径：`~/.cache/openpi`

### 5.5 机器人控制器

**文件**: `robot_zero.py`

ROS2 关节插值控制器：
- 100Hz 平滑轨迹插值
- 支持左右臂独立控制
- 收敛检测（容差和持续时间检查）
- 配置驱动的关节限位

---

## 6. 新增配置文件

### 6.1 机器人运动学配置

**文件**: `arm_config.yml`

```yaml
# 内容包括：
- 关节限位（上下界）
- 速度和加速度限制（7 DOF）
- IK 求解器配置
- 坐标变换矩阵（T_lee2pee）
- 反馈控制增益（Kp, alpha）
```

---

## 7. 依赖修改

**文件**: `pyproject.toml`

新增依赖：
```toml
"google-cloud-storage>=3.1.0"  # GCS 存储桶访问
```

---

## 8. 训练产物

### 8.1 模型资产

| 目录 | 说明 |
|------|------|
| `assets/pi0_teleavatar/` | Teleavatar 模型资产（norm_stats 等）|
| `assets/pi0_teleavatar_lora/` | LoRA 微调变体资产 |

### 8.2 检查点

- `checkpoints/` - 训练检查点目录

### 8.3 训练日志

- `wandb/` - Weights & Biases 训练记录（14+ 次训练运行）

---

## 9. 使用指南

### 9.1 训练 Teleavatar 模型

```bash
# 1. 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar

# 2. 启动训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar \
    --exp-name=my_experiment \
    --overwrite
```

### 9.2 低显存训练（LoRA）

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar_lora
uv run scripts/train.py pi0_teleavatar_lora --exp-name=lora_experiment
```

### 9.3 启动策略服务器

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar \
    --policy.dir=checkpoints/pi0_teleavatar/my_experiment/20000
```

### 9.4 数据转换

```bash
uv run examples/teleavatar/convert_teleavatar_data_to_lerobot.py \
    --data_dir /path/to/teleavatar/data
```

---

## 10. 文件变更清单

### 新增文件

```
src/openpi/policies/teleavatar_policy.py
src/openpi/policies/teleavatar_policy_endeffector.py
examples/teleavatar/  (13 files)
scripts/train_pick_up_paper.sh
scripts/merge_lerobot_datasets.py
download_pi0.py
robot_zero.py
arm_config.yml
docs/MODIFICATIONS.md
CLAUDE.md
```

### 修改文件

```
src/openpi/models/pi0.py
src/openpi/models/pi0_config.py
src/openpi/training/config.py
src/openpi/training/data_loader.py
scripts/compute_norm_stats.py
pyproject.toml
examples/convert_jax_model_to_pytorch.py
```

### 删除文件

```
src/openpi/training/misc/polaris_config.py
```
