import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05

        # Soft Inpainting configuration
        self.use_soft_inpainting = config.use_soft_inpainting
        self.time_threshold_inpaint = config.time_threshold_inpaint
        self.use_correlated_noise = config.use_correlated_noise
        self.correlation_beta = config.correlation_beta

        # Correlation matrix (lazy loaded)
        self.correlation_loaded = False
        self.action_correlation_cholesky = None
        self.inpainting_cache = {}
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def load_correlation_matrix(self, norm_stats: dict) -> None:
        """Load correlation matrix from norm_stats and apply beta shrinkage.

        Args:
            norm_stats: Dictionary containing 'actions' key with action_correlation_cholesky field.
        """
        if not self.use_correlated_noise:
            logger.info("Correlated noise disabled in config, skipping correlation matrix loading")
            return

        actions_stats = norm_stats.get("actions")
        if actions_stats is None:
            logger.warning("No 'actions' key in norm_stats, skipping correlation matrix loading")
            return

        # Extract correlation matrix (support both dict and attribute access)
        chol_matrix = None
        if isinstance(actions_stats, dict):
            chol_matrix = actions_stats.get("action_correlation_cholesky")
        elif hasattr(actions_stats, "action_correlation_cholesky"):
            chol_matrix = actions_stats.action_correlation_cholesky

        if chol_matrix is None:
            logger.warning(
                "action_correlation_cholesky not found in norm_stats['actions']. "
                "Run compute_correlation_matrix.py to generate it."
            )
            return

        # Convert to jax array
        L = jnp.asarray(chol_matrix)

        # Validate shape
        expected_dim = self.action_horizon * self.action_dim
        if L.shape[0] != expected_dim or L.shape[1] != expected_dim:
            raise ValueError(
                f"Correlation matrix has wrong dimensions: {L.shape}. "
                f"Expected ({expected_dim}, {expected_dim})."
            )

        # Apply beta shrinkage: L_reg = beta * L + (1-beta) * I
        beta = self.correlation_beta
        L_reg = beta * L + (1 - beta) * jnp.eye(L.shape[0])

        self.action_correlation_cholesky = L_reg
        self.correlation_loaded = True
        logger.info(
            f"Loaded correlation matrix with shape {L_reg.shape}, beta={beta}"
        )

    def _precompute_correction_matrix(
        self,
        O_indices: at.Int[at.Array, " nO"],
        U_indices: at.Int[at.Array, " nU"],
    ) -> dict:
        """Precompute matrix for correlation-aware inpainting correction.

        Computes Sigma_UO @ Sigma_OO^{-1} which propagates corrections from O to U
        while preserving correlation structure.

        Args:
            O_indices: Flat indices of inpainted dimensions [|O|]
            U_indices: Flat indices of free dimensions [|U|]

        Returns:
            Dictionary with {O_indices, U_indices, correction_matrix}
        """
        if not self.correlation_loaded:
            raise RuntimeError(
                "Cannot precompute correction matrix: correlation matrix not loaded. "
                "Call load_correlation_matrix() first."
            )

        L = self.action_correlation_cholesky
        Sigma = L @ L.T  # Full covariance matrix

        # Extract submatrices
        Sigma_OO = Sigma[jnp.ix_(O_indices, O_indices)]
        Sigma_UO = Sigma[jnp.ix_(U_indices, O_indices)]

        # Regularize and solve
        eps = 1e-6 * jnp.maximum(jnp.mean(jnp.diag(Sigma_OO)), 1.0)
        Sigma_OO_reg = Sigma_OO + eps * jnp.eye(Sigma_OO.shape[0])

        # Compute correction matrix: Sigma_UO @ Sigma_OO^{-1}
        correction_matrix = jax.scipy.linalg.solve(
            Sigma_OO_reg, Sigma_UO.T, assume_a="pos"
        ).T

        return {
            "O_indices": O_indices,
            "U_indices": U_indices,
            "correction_matrix": correction_matrix,
        }

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        initial_actions: at.Float[at.Array, "b n ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ===== Soft Inpainting initialization =====
        fixed_z_O = None
        x0_O = None
        O_indices = None
        U_indices = None
        inpainting_cache = None

        if initial_actions is not None and self.use_soft_inpainting:
            num_initial = initial_actions.shape[1]
            input_dim = initial_actions.shape[2]
            flat_dim = self.action_horizon * self.action_dim

            # Compute O_indices (inpainted dimensions) and U_indices (free dimensions)
            O_indices = jnp.array([
                t * self.action_dim + d
                for t in range(num_initial)
                for d in range(input_dim)
            ], dtype=jnp.int32)

            O_set = {t * self.action_dim + d for t in range(num_initial) for d in range(input_dim)}
            U_indices = jnp.array([
                i for i in range(flat_dim) if i not in O_set
            ], dtype=jnp.int32)

            # Pad initial_actions to full model dimensions
            if input_dim < self.action_dim:
                padding = jnp.zeros((batch_size, num_initial, self.action_dim - input_dim))
                initial_actions_padded = jnp.concatenate([initial_actions, padding], axis=2)
            else:
                initial_actions_padded = initial_actions[:, :, :self.action_dim]

            if num_initial < self.action_horizon:
                seq_padding = jnp.zeros((batch_size, self.action_horizon - num_initial, self.action_dim))
                initial_actions_padded = jnp.concatenate([initial_actions_padded, seq_padding], axis=1)
            else:
                initial_actions_padded = initial_actions_padded[:, :self.action_horizon]

            # Extract fixed_z_O and x0_O
            noise_flat = noise.reshape(batch_size, flat_dim)
            fixed_z_O = noise_flat[:, O_indices]

            x0_O = initial_actions_padded.reshape(batch_size, flat_dim)[:, O_indices]

            # Precompute correction matrix if correlation is loaded
            if self.correlation_loaded:
                cache_key = (num_initial, input_dim)
                if cache_key not in self.inpainting_cache:
                    logger.info(f"Computing correction matrix for {num_initial} steps, {input_dim} dims...")
                    self.inpainting_cache[cache_key] = self._precompute_correction_matrix(O_indices, U_indices)
                inpainting_cache = self.inpainting_cache[cache_key]

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Store references for use in step function
        time_threshold = self.time_threshold_inpaint

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Euler step
            x_t_new = x_t + dt * v_t

            # ===== Soft Inpainting constraint =====
            if fixed_z_O is not None:
                time_new = time + dt

                def apply_correlated_correction(x):
                    x_flat = x.reshape(batch_size, -1)

                    # Compute desired state at O: x_t[O] = (1-t)*x0[O] + t*z_O
                    x_desired_O = (1.0 - time_new) * x0_O + time_new * fixed_z_O

                    # Compute correction at O
                    delta_O = x_desired_O - x_flat[:, O_indices]

                    # Apply hard constraint at O
                    x_flat = x_flat.at[:, O_indices].set(x_desired_O)

                    # If correlation matrix available, propagate correction to U
                    if inpainting_cache is not None:
                        correction_matrix = inpainting_cache["correction_matrix"]
                        U_indices_cached = inpainting_cache["U_indices"]

                        # Compute correlated correction: delta_U = delta_O @ correction_matrix.T
                        delta_U = delta_O @ correction_matrix.T

                        # Skip if correction too large (indicates instability)
                        max_correction = jnp.max(jnp.abs(delta_U))
                        x_flat = jax.lax.cond(
                            max_correction <= 1.0,
                            lambda xf: xf.at[:, U_indices_cached].add(delta_U),
                            lambda xf: xf,
                            x_flat
                        )

                    return x_flat.reshape(batch_size, self.action_horizon, self.action_dim)

                # Only apply correction when time > threshold (soft constraint in final steps)
                x_t_new = jax.lax.cond(
                    time_new > time_threshold,
                    apply_correlated_correction,
                    lambda x: x,
                    x_t_new
                )

            return x_t_new, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
