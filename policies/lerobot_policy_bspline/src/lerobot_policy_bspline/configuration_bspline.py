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
        bspline_mode: ``diffusion`` (B-spline Policy, the options above), ``onestep`` (one-step regression
            of fine control points on a fixed knot grid) or ``planbind`` (a coarse plan, then a residual on
            its exact lift; one shared UNet). The last two use the `planbind_*` options and ignore
            `bspline_chunk_size`, `bspline_max_error`, `bspline_speed_up`, `bspline_fit_workers` and the
            diffusion sampling options; `bspline_window` sets the frames one chunk covers.
        planbind_fine_spacing: Frames between fine knots. The network is called once per fine span.
        planbind_subdivision: Fine spans per coarse span (coarse spacing = fine spacing * subdivision).
        planbind_max_plan_age: Oldest plan, in frames, that Bind is trained on and run with; older plans
            are replaced. A multiple of `planbind_fine_spacing`.
        planbind_plan_noise: Std of the noise added to the stale plans Bind is trained on.
        planbind_replan_threshold: Re-plan when Bind's largest residual exceeds this (normalized action
            units); 0 disables it.
        planbind_latency: Simulated inference latency at evaluation, in frames: a call's result is used
            from this many frames after its observation.
        planbind_commit: Control points kept from the running chunk at each call: ``latency`` (everything
            up to the swap, exact C2 join), ``now`` (C2 join at the call frame only) or ``none`` (naive swap).
        planbind_position_limit: Bound on the normalized actions enforced on the control points; 0 is off.
        planbind_velocity_limit: Bound on the normalized action change per frame; 0 is off.
        planbind_acceleration_limit: Bound on the normalized action change per frame squared; 0 is off.
            At a position limit, the position and velocity bounds win and acceleration may exceed it.
        planbind_onestep_input: `onestep` only. ``zeros`` feeds the UNet a constant sequence (D1's
            control); ``learned`` feeds a trained ``(n_fine, action_dim)`` input initialised to a linear ramp,
            so the input carries sequence position.
        planbind_call_every: Frames between network calls at evaluation, a multiple of
            `planbind_fine_spacing`; 0 calls once per fine span. A call never starts while one is pending.
        bspline_runtime: Evaluation only, `diffusion` mode: run the policy through the same runtime as
            `onestep` and `planbind` (latency, commit, call rate), refitting each denoised spline to the fine
            knot grid. Off keeps B-spline Policy's own action queue.
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

    bspline_mode: str = "diffusion"
    planbind_fine_spacing: int = 4
    planbind_subdivision: int = 4
    planbind_max_plan_age: int = 32
    planbind_plan_noise: float = 0.02
    planbind_replan_threshold: float = 0.0
    planbind_latency: int = 0
    planbind_commit: str = "latency"
    planbind_position_limit: float = 0.0
    planbind_velocity_limit: float = 0.0
    planbind_acceleration_limit: float = 0.0
    planbind_call_every: int = 0
    planbind_onestep_input: str = "zeros"
    bspline_runtime: bool = False

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
        if self.bspline_mode not in ("diffusion", "onestep", "planbind"):
            raise ValueError(
                f"`bspline_mode` must be 'diffusion', 'onestep' or 'planbind'. Got {self.bspline_mode!r}."
            )
        if self.planbind_onestep_input not in ("zeros", "learned"):
            raise ValueError(
                f"`planbind_onestep_input` must be 'zeros' or 'learned'. Got {self.planbind_onestep_input!r}."
            )
        if self.planbind_onestep_input == "learned" and self.bspline_mode != "onestep":
            raise ValueError(
                "`planbind_onestep_input='learned'` only applies to `bspline_mode='onestep'`. "
                f"Got {self.bspline_mode!r}."
            )
        if self.bspline_mode != "diffusion" or self.bspline_runtime:
            self._validate_planbind()

    def _validate_planbind(self) -> None:
        if self.bspline_degree != 3 and self.bspline_mode != "diffusion":
            raise ValueError("`onestep` and `planbind` use cubic splines: set `bspline_degree=3`.")
        if self.planbind_fine_spacing < 1 or self.planbind_subdivision < 1:
            raise ValueError("`planbind_fine_spacing` and `planbind_subdivision` must be >= 1.")
        if self.planbind_max_plan_age < 0 or self.planbind_max_plan_age % self.planbind_fine_spacing:
            raise ValueError(
                "`planbind_max_plan_age` must be a non-negative multiple of `planbind_fine_spacing`. "
                f"Got {self.planbind_max_plan_age} and {self.planbind_fine_spacing}."
            )
        if self.planbind_latency < 0:
            raise ValueError(f"`planbind_latency` must be >= 0. Got {self.planbind_latency}.")
        if self.planbind_call_every < 0 or self.planbind_call_every % self.planbind_fine_spacing:
            raise ValueError(
                "`planbind_call_every` must be 0 or a positive multiple of `planbind_fine_spacing`. "
                f"Got {self.planbind_call_every} and {self.planbind_fine_spacing}."
            )
        if self.planbind_latency + self.call_period >= self.bspline_window:
            raise ValueError(
                "`planbind_latency` plus the call period (`planbind_call_every`) must be below `bspline_window`."
            )
        if self.planbind_commit not in ("latency", "now", "none"):
            raise ValueError(
                f"`planbind_commit` must be 'latency', 'now' or 'none'. Got {self.planbind_commit!r}."
            )
        for name in (
            "planbind_plan_noise",
            "planbind_replan_threshold",
            "planbind_position_limit",
            "planbind_velocity_limit",
            "planbind_acceleration_limit",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be >= 0. Got {getattr(self, name)}.")

    @property
    def call_period(self) -> int:
        """Frames between network calls in the spline runtime."""
        return self.planbind_call_every or self.planbind_fine_spacing

    @property
    def planbind_offset(self) -> int:
        """Frames of action history loaded before the current frame (stale plans for Bind training)."""
        return self.planbind_max_plan_age if self.bspline_mode == "planbind" else 0

    @property
    def bspline_time_scale(self) -> float:
        """Knot times in ``[0, bspline_time_scale]`` frames are normalized to ``[-1, 1]``."""
        return float(self.bspline_window - 1)

    @property
    def action_delta_indices(self) -> list:
        """Raw action window loaded per sample: the current frame and the next ones.

        `planbind` also loads `planbind_max_plan_age` earlier frames for the stale plans.
        """
        return list(range(-self.planbind_offset, self.bspline_window))
