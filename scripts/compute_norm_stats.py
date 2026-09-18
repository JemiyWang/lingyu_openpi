"""Compute normalization statistics for a config.

This script computes:
1. Normalization statistics (mean, std, quantiles) for state and actions
2. Optionally: correlation matrix for Soft Inpainting (--compute-correlation)
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_correlation_matrix(
    action_chunks: np.ndarray,
    max_samples: int = 2000000,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Cholesky decomposition of correlation matrix from action chunks.

    Args:
        action_chunks: Array of shape (N, action_horizon, action_dim)
        max_samples: Maximum samples to use (for memory efficiency)
        epsilon: Regularization for numerical stability

    Returns:
        Cholesky decomposition of correlation matrix, shape (H*D, H*D)
    """
    num_samples, action_horizon, action_dim = action_chunks.shape
    flat_dim = action_horizon * action_dim

    print(f"Computing correlation matrix: {num_samples} samples, flat_dim={flat_dim}")

    # Subsample if needed
    if num_samples > max_samples:
        print(f"Subsampling from {num_samples} to {max_samples} samples")
        rng = np.random.RandomState(42)
        indices = rng.choice(num_samples, size=max_samples, replace=False)
        action_chunks = action_chunks[indices]
        num_samples = max_samples

    # Flatten to (N, H*D)
    flattened = action_chunks.reshape(num_samples, flat_dim)

    # Normalize to zero mean, unit variance
    mean = np.mean(flattened, axis=0)
    std = np.std(flattened, axis=0)

    # Handle constant dimensions
    constant_dims = std < 1e-6
    print(f"Found {np.sum(constant_dims)} constant dimensions")

    std_safe = std.copy()
    std_safe[constant_dims] = 1.0
    normalized = (flattened - mean) / std_safe
    normalized[:, constant_dims] = 0.0

    # Compute covariance matrix
    print("Computing covariance matrix...")
    cov_matrix = np.cov(normalized, rowvar=False)

    # Enforce diagonal = 1
    diag_vals = np.diag(cov_matrix).copy()
    diag_vals[diag_vals < 1e-10] = 1.0
    normalizer = np.sqrt(diag_vals[:, None] @ diag_vals[None, :])
    cov_matrix = cov_matrix / normalizer

    # Handle constant dimensions
    cov_matrix[constant_dims, :] = 0.0
    cov_matrix[:, constant_dims] = 0.0
    np.fill_diagonal(cov_matrix, 1.0)

    # Add regularization
    cov_matrix_reg = cov_matrix + epsilon * np.eye(flat_dim)

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(cov_matrix_reg)
    min_eigenvalue = np.min(eigenvalues)
    print(f"Min eigenvalue: {min_eigenvalue:.6e}")

    if min_eigenvalue <= 0:
        print("Matrix not positive definite, adding stronger regularization")
        epsilon = max(1e-5, -min_eigenvalue + 1e-5)
        cov_matrix_reg = cov_matrix + epsilon * np.eye(flat_dim)

    # Cholesky decomposition
    print("Computing Cholesky decomposition...")
    chol_lower = np.linalg.cholesky(cov_matrix_reg)
    print(f"Cholesky decomposition successful, shape: {chol_lower.shape}")

    return chol_lower


def main(
    config_name: str,
    max_frames: int | None = None,
    compute_correlation: bool = False,
    max_correlation_samples: int = 2000000,
):
    """Compute normalization statistics for a config.

    Args:
        config_name: Name of the training config (e.g., "pi05_teleavatar")
        max_frames: Maximum number of frames to use (None = use all)
        compute_correlation: Whether to compute correlation matrix for Soft Inpainting
        max_correlation_samples: Maximum samples for correlation matrix computation
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # Collect action chunks for correlation matrix if requested
    all_action_chunks = [] if compute_correlation else None

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

        # Collect action chunks for correlation matrix
        if compute_correlation:
            actions = np.asarray(batch["actions"])
            # actions shape: (batch_size, action_horizon, action_dim)
            all_action_chunks.append(actions)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Compute and save correlation matrix if requested
    if compute_correlation and all_action_chunks:
        print("\n" + "=" * 60)
        print("Computing correlation matrix for Soft Inpainting...")
        print("=" * 60)

        action_chunks = np.concatenate(all_action_chunks, axis=0)
        print(f"Total action chunks: {action_chunks.shape}")

        chol_matrix = compute_correlation_matrix(
            action_chunks,
            max_samples=max_correlation_samples,
        )

        # Save correlation matrix
        output_path.mkdir(parents=True, exist_ok=True)
        chol_path = output_path / "action_correlation_cholesky.npy"
        np.save(chol_path, chol_matrix)
        print(f"Saved correlation matrix to: {chol_path}")
        print(f"Matrix shape: {chol_matrix.shape}")
        print(f"Memory size: {chol_matrix.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    tyro.cli(main)
