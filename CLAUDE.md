# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPI is Physical Intelligence's open-source repository for robotics models, containing π₀, π₀-FAST, and π₀.₅ vision-language-action (VLA) models. These are flow-based and autoregressive models for robot control, pre-trained on 10k+ hours of robot data.

## Common Commands

### Environment Setup
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
git submodule update --init --recursive  # if already cloned

# Install dependencies (uses uv package manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Training
```bash
# Compute normalization stats (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# JAX training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name> --overwrite

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <name>

# PyTorch training (multi-GPU)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <name>
```

### Inference
```bash
# Start policy server
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
```

### Code Quality
```bash
# Setup pre-commit hooks (first time only)
pre-commit install

# Linting and formatting (run before commits)
ruff check .
ruff format .

# Run tests (searches src/, scripts/, packages/)
uv run pytest

# Run single test file
uv run pytest <path_to_test.py>

# Run specific test by name
uv run pytest -k "test_name"

# Run manual tests (marked with @pytest.mark.manual)
uv run pytest -m manual
```

## Architecture

### Source Code (`src/openpi/`)

**models/** - VLA model implementations (JAX primary)
- `pi0.py` - π₀ flow-matching model
- `pi0_fast.py` - π₀-FAST autoregressive model
- `gemma.py` - Vision encoder (SigLIP) + language model (Gemma) backbone
- `lora.py` - Low-rank adaptation for efficient fine-tuning

**models_pytorch/** - PyTorch equivalents for distributed training

**policies/** - Robot-agnostic inference wrappers
- `policy.py` - Core `Policy` class handling input/output transforms and model inference
- `policy_config.py` - `create_trained_policy()` factory function
- Robot-specific policies: `aloha_policy.py`, `droid_policy.py`, `libero_policy.py`, `teleavatar_policy.py`

**training/** - Training infrastructure
- `config.py` - Centralized config registry with all preset training configs
- `data_loader.py` - Data loading (LeRobot and RLDS formats)
- `weight_loaders.py` - Loading pretrained weights from base models

**transforms.py** - Composable data transformation pipeline (`DataTransformFn` protocol)

**serving/** - WebSocket policy server for remote inference

### Data Flow

```
Robot Observations → Data Transforms → Model.sample_actions() → Output Transforms → Robot Actions
```

Key pattern: All data preprocessing is composable transforms chained together. Robot-specific policies define their own `Inputs`/`Outputs` dataclasses that map robot observations to model inputs.

### Configuration System

Training configs are registered in `src/openpi/training/config.py`. Key preset configs:
- Base models: `pi0_base`, `pi0_fast_base`, `pi05_base`
- Fine-tuned: `pi05_droid`, `pi05_libero`, `pi0_aloha_sim`
- Debug: `debug`, `debug_pi05` (fast iteration)

### Examples (`examples/`)

Each subdirectory contains robot-specific integration:
- `aloha_sim/`, `aloha_real/` - ALOHA platform
- `droid/` - DROID platform with RLDS dataset support
- `libero/` - LIBERO benchmark
- `teleavatar/` - Teleavatar dual-arm robot with ROS2 integration
- `simple_client/` - Minimal WebSocket client example

### PyTorch Setup

For PyTorch models, apply transformers patches after `uv sync`:
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Key Patterns

- **Dual framework support**: JAX (primary training) and PyTorch (distributed training with DDP)
- **Composable transforms**: `DataTransformFn` protocol enables flexible preprocessing pipelines
- **Config-driven**: Switch robots/datasets by changing config name, not code
- **Remote inference**: Policy server runs on GPU machine, robot connects via WebSocket

## Teleavatar Extensions

This fork adds Teleavatar dual-arm robot support. See [docs/MODIFICATIONS.md](docs/MODIFICATIONS.md) for full details.

**New policies**: `teleavatar_policy.py`, `teleavatar_policy_endeffector.py`

**New training configs**:
- `pi0_teleavatar` - Standard fine-tuning with soft inpainting
- `pi0_teleavatar_lora` - LoRA fine-tuning (16GB GPU)
- `pi0_teleavatar_endeffector` - End-effector representation

**Soft Inpainting**: Enhanced `pi0.py` with correlation-aware action constraints. Config params in `pi0_config.py`:
- `use_soft_inpainting`, `time_threshold_inpaint`, `use_correlated_noise`, `correlation_beta`

**Inference Recording**: Record data during model inference for analysis or dataset expansion. See [docs/inference_recording.md](docs/inference_recording.md).

```bash
# Run with data recording enabled
python examples/teleavatar/main.py --remote-host 127.0.0.1 --record

# Keyboard controls during recording:
# 'r' - Start recording
# 'q' - Stop recording
# 's' - Save as success
# 'f' - Save as failure
# 'd' - Discard
```

**Example usage**:
```bash
# Train Teleavatar model
uv run scripts/compute_norm_stats.py --config-name pi0_teleavatar
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_teleavatar --exp-name=my_exp

# Serve policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_teleavatar --policy.dir=checkpoints/pi0_teleavatar/my_exp/20000
```
