"""Configuration for B-spline Policy (arXiv 2607.09648) on top of LeRobot's diffusion policy."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("bspline")
@dataclass
class BsplineConfig(DiffusionConfig):
    """Diffusion policy whose denoised sample is a B-spline parameter chunk instead of raw actions.

    The diffusion UNet denoises a ``(horizon, 1 + action_dim)`` matrix: one knot column plus
    `action_dim` control-point columns, with ``horizon = bspline_chunk_size + 2 * bspline_degree``.
    During training each target is fitted from a raw window of `bspline_window` future actions
    (starting at the current frame). At inference the predicted spline is sampled once per control
    step, `bspline_speed_up` frames apart.

    Args:
        horizon: Rows of the B-spline parameter matrix. Must equal
            ``bspline_chunk_size + 2 * bspline_degree`` and divide by ``2 ** len(down_dims)``.
        n_action_steps: Maximum number of actions decoded from one predicted spline. Fewer are
            queued when the predicted spline ends earlier.
        bspline_degree: Spline degree (3 = cubic, as upstream).
        bspline_chunk_size: Knot intervals held by one parameter chunk (upstream `chunk_size`).
        bspline_window: Raw future actions loaded per training sample, in frames. It bounds the time
            a single chunk can cover and sets the knot normalization range ``[0, bspline_window - 1]``.
        bspline_max_error: Max abs fitting error, in normalized action units, when choosing knots.
        bspline_speed_up: Frames of spline time advanced per control step at inference. Values above
            1 execute the demonstrated motion faster (upstream `speed_up_times`).
        bspline_fit_workers: Processes that fit the training windows in parallel; 0 fits them in the
            training process. Under multi-GPU training every GPU process starts its own pool.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 32
    drop_n_last_frames: int = 0

    # Control points of min-max normalized actions can overshoot [-1, 1] slightly.
    clip_sample_range: float = 1.5

    bspline_degree: int = 3
    bspline_chunk_size: int = 10
    bspline_window: int = 64
    bspline_max_error: float = 0.01
    bspline_speed_up: float = 1.0
    bspline_fit_workers: int = 0

    def __post_init__(self):
        """Validate the diffusion options, then the B-spline options."""
        super().__post_init__()
        if self.bspline_degree < 1:
            raise ValueError(f"`bspline_degree` must be >= 1. Got {self.bspline_degree}.")
        if self.bspline_chunk_size < 2:
            raise ValueError(f"`bspline_chunk_size` must be >= 2. Got {self.bspline_chunk_size}.")
        expected_horizon = self.bspline_chunk_size + 2 * self.bspline_degree
        if self.horizon != expected_horizon:
            raise ValueError(
                "`horizon` is the number of B-spline parameter rows and must equal "
                f"`bspline_chunk_size + 2 * bspline_degree` = {expected_horizon}. Got {self.horizon}."
            )
        if self.bspline_window < self.bspline_degree + 1:
            raise ValueError(f"`bspline_window` must be >= bspline_degree + 1. Got {self.bspline_window}.")
        if self.bspline_max_error <= 0:
            raise ValueError(f"`bspline_max_error` must be > 0. Got {self.bspline_max_error}.")
        if self.bspline_speed_up <= 0:
            raise ValueError(f"`bspline_speed_up` must be > 0. Got {self.bspline_speed_up}.")
        if self.bspline_fit_workers < 0:
            raise ValueError(f"`bspline_fit_workers` must be >= 0. Got {self.bspline_fit_workers}.")
        if self.do_mask_loss_for_padding:
            raise ValueError(
                "`do_mask_loss_for_padding` is not supported: padded frames are dropped before the "
                "spline fit, so there is no per-row padding mask."
            )

    @property
    def bspline_time_scale(self) -> float:
        """Knot times in ``[0, bspline_time_scale]`` frames are normalized to ``[-1, 1]``."""
        return float(self.bspline_window - 1)

    @property
    def action_delta_indices(self) -> list:
        """Raw action window loaded per sample: the current frame and the next ones."""
        return list(range(self.bspline_window))
