"""One-step and Plan-Bind spline heads on a fixed knot grid (`bspline_mode` = `onestep` / `planbind`).

Both modes regress control points with LeRobot's diffusion UNet used as a one-step network: the input
sequence is the "sample" and the diffusion timestep embedding carries the call's role.

- `onestep` (control): the UNet maps zeros to the fine control points of the next window.
- `planbind`: the same UNet first maps zeros to a coarse plan (timestep 0). The plan is lifted exactly to
  the fine grid and shifted to the current frame; the UNet then maps that lifted plan to a residual
  (timestep ``1 + plan age``). One shared UNet keeps the parameter count equal to `onestep`.

`SplineRuntime` runs either mode inside `select_action`: it calls the network every `call_period` frames
(one fine knot span by default), simulates an inference latency (the result of a call issued at frame k
swaps in at k + latency, and calls do not overlap), freezes the control points that the commit rule keeps,
and projects the free ones. With `bspline_runtime` it also runs `diffusion` mode, whose denoised spline is
refitted to the fine knot grid at each call.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.utils.constants import ACTION

from .configuration_bspline import BsplineConfig
from .uniform_spline import basis_matrix, lift_matrix, n_ctrl_points, n_frozen, project, shift_ctrl


def knot_grid(config: BsplineConfig) -> tuple[int, int]:
    """Coarse and fine control points of one `bspline_window` on the fixed knot grid."""
    n_coarse = n_ctrl_points(
        config.bspline_window, config.planbind_fine_spacing * config.planbind_subdivision
    )
    n_fine = 3 + (n_coarse - 3) * config.planbind_subdivision
    assert n_fine >= n_ctrl_points(config.bspline_window, config.planbind_fine_spacing)
    return n_coarse, n_fine


def register_fine_grid(module: torch.nn.Module, config: BsplineConfig) -> Tensor:
    """Give `module` the fine basis ``(window, n_fine)`` and its least-squares fit; returns the basis."""
    frames = torch.arange(config.bspline_window, dtype=torch.float64)
    fine_basis = basis_matrix(frames, knot_grid(config)[1], config.planbind_fine_spacing)
    # Derived from the config, so kept out of checkpoints.
    module.register_buffer("fine_basis", fine_basis.float(), persistent=False)
    module.register_buffer("fine_fit", torch.linalg.pinv(fine_basis).float(), persistent=False)
    return fine_basis


class PlanBindModel(DiffusionModel):
    """Observation encoders and UNet of the diffusion model, trained as one-step spline regressors."""

    def __init__(self, config: BsplineConfig):
        """Build the diffusion modules and the fixed spline matrices for `config`."""
        super().__init__(config)
        self.fine_spacing = config.planbind_fine_spacing
        self.subdivision = config.planbind_subdivision
        self.n_coarse, self.n_fine = knot_grid(config)
        frames = torch.arange(config.bspline_window, dtype=torch.float64)
        register_fine_grid(self, config)
        coarse_basis = basis_matrix(frames, self.n_coarse, self.fine_spacing * self.subdivision)
        self.register_buffer("coarse_basis", coarse_basis.float(), persistent=False)
        self.register_buffer("coarse_fit", torch.linalg.pinv(coarse_basis).float(), persistent=False)
        lift = lift_matrix(self.n_coarse, self.n_fine, self.subdivision)
        self.register_buffer("lift", lift.float(), persistent=False)
        self.row_multiple = 2 ** len(config.down_dims)

    def _run_unet(self, x: Tensor, role: Tensor, global_cond: Tensor) -> Tensor:
        """UNet on ``(B, N, D)``, rows padded by repeating the last one to the UNet's length multiple."""
        n = x.shape[1]
        pad = -n % self.row_multiple
        if pad:
            x = torch.cat([x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        return self.unet(x, role, global_cond=global_cond)[:, :n]

    def plan(self, global_cond: Tensor) -> Tensor:
        """Coarse control points ``(B, n_coarse, D)`` of the next window."""
        zeros = global_cond.new_zeros(
            global_cond.shape[0], self.n_coarse, self.config.action_feature.shape[0]
        )
        role = torch.zeros(global_cond.shape[0], dtype=torch.long, device=global_cond.device)
        return self._run_unet(zeros, role, global_cond)

    def onestep(self, global_cond: Tensor) -> Tensor:
        """Fine control points ``(B, n_fine, D)`` of the next window, without a plan."""
        zeros = global_cond.new_zeros(global_cond.shape[0], self.n_fine, self.config.action_feature.shape[0])
        role = torch.zeros(global_cond.shape[0], dtype=torch.long, device=global_cond.device)
        return self._run_unet(zeros, role, global_cond)

    def lifted_plan(self, coarse: Tensor, age_spans: Tensor) -> Tensor:
        """Coarse plans lifted to the fine grid and re-anchored ``age_spans`` fine spans later."""
        return shift_ctrl(torch.einsum("fc,bcd->bfd", self.lift, coarse), age_spans)

    def bind(self, lifted: Tensor, age_frames: Tensor, global_cond: Tensor) -> Tensor:
        """Residual ``(B, n_fine, D)`` added to the lifted plan."""
        return self._run_unet(lifted, 1 + age_frames.long(), global_cond)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Masked MSE of the decoded trajectories against the current action window.

        ``batch[ACTION]`` is ``(B, max_age + window, D)``: the window starts ``max_age`` frames in, and the
        frames before it give the stale plans Bind is trained on.
        """
        cfg = self.config
        window = cfg.bspline_window
        offset = cfg.planbind_offset
        actions = batch[ACTION]
        is_pad = batch.get("action_is_pad")
        current = actions[:, offset : offset + window]
        mask = torch.ones(current.shape[:2], device=current.device)
        if is_pad is not None:
            mask = (~is_pad[:, offset : offset + window]).float()
        global_cond = self._prepare_global_conditioning(batch)

        def traj_loss(basis: Tensor, ctrl: Tensor) -> Tensor:
            err = F.mse_loss(torch.einsum("tn,bnd->btd", basis, ctrl), current, reduction="none").mean(-1)
            return (err * mask).sum() / mask.sum().clamp_min(1)

        if cfg.bspline_mode == "onestep":
            return traj_loss(self.fine_basis, self.onestep(global_cond))

        plan_loss = traj_loss(self.coarse_basis, self.plan(global_cond))
        # Bind sees a plan made `age` frames ago (fitted to the demonstration there), plus noise.
        batch_size = actions.shape[0]
        n_ages = cfg.planbind_max_plan_age // self.fine_spacing + 1
        age_spans = torch.randint(0, n_ages, (batch_size,), device=actions.device)
        age_frames = age_spans * self.fine_spacing
        starts = offset - age_frames
        idx = starts[:, None] + torch.arange(window, device=actions.device)[None, :]
        stale = torch.gather(actions, 1, idx[..., None].expand(-1, -1, actions.shape[2]))
        coarse = torch.einsum("cw,bwd->bcd", self.coarse_fit, stale)
        coarse = coarse + cfg.planbind_plan_noise * torch.randn_like(coarse)
        lifted = self.lifted_plan(coarse, age_spans)
        bind_loss = traj_loss(self.fine_basis, lifted + self.bind(lifted, age_frames, global_cond))
        return plan_loss + bind_loss


class SplineRuntime:
    """Stateful execution of fine control-point chunks for a batch of environments."""

    def __init__(
        self,
        model: DiffusionModel,
        config: BsplineConfig,
        chunk_fn: Callable[[Tensor], Tensor] | None = None,
    ):
        """Run `model` with the runtime options of `config`.

        `model` holds the fine grid (`register_fine_grid`). `chunk_fn` maps observation features to fine
        control points ``(B, n_fine, D)`` anchored at the call frame; None calls `model` in its own mode.
        """
        self.model = model
        self.config = config
        self.chunk_fn = chunk_fn
        self.reset()

    def reset(self) -> None:
        """Forget the running chunk, the plan and any pending call (new episode)."""
        self.frame = 0
        self.cmd: Tensor | None = None  # (B, n_fine, D), anchored at `cmd_anchor`
        self.cmd_anchor = 0
        self.plan: Tensor | None = None  # (B, n_coarse, D)
        self.plan_anchor: Tensor | None = None  # (B,) frame each plan was made at
        self.pending: tuple[int, Tensor] | None = None  # (swap frame, chunk anchored at its call frame)
        self.pending_anchor = 0
        self.last_residual: Tensor | None = None

    def _call(self, global_cond: Tensor) -> Tensor:
        """Fine control points anchored at the current frame (before commit and projection)."""
        if self.chunk_fn is not None:
            return self.chunk_fn(global_cond)
        model, cfg = self.model, self.config
        assert isinstance(model, PlanBindModel)
        if cfg.bspline_mode == "onestep":
            return model.onestep(global_cond)
        batch_size = global_cond.shape[0]
        frame = torch.full((batch_size,), self.frame, device=global_cond.device)
        if self.plan is None:
            self.plan = model.plan(global_cond)
            self.plan_anchor = frame.clone()
        assert self.plan_anchor is not None
        # Re-plan where the plan has aged past what Bind was trained on.
        expired = frame - self.plan_anchor > cfg.planbind_max_plan_age
        if expired.any():
            self._replan(global_cond, expired, frame)
        chunk, residual = self._bind(global_cond, frame)
        if cfg.planbind_replan_threshold > 0:
            surprised = (residual > cfg.planbind_replan_threshold) & (self.plan_anchor < frame)
            if surprised.any():
                self._replan(global_cond, surprised, frame)
                chunk, residual = self._bind(global_cond, frame)
        self.last_residual = residual
        return chunk

    def _replan(self, global_cond: Tensor, where: Tensor, frame: Tensor) -> None:
        assert self.plan is not None and self.plan_anchor is not None
        fresh = self.model.plan(global_cond)
        self.plan = torch.where(where[:, None, None], fresh, self.plan)
        self.plan_anchor = torch.where(where, frame, self.plan_anchor)

    def _bind(self, global_cond: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
        assert self.plan is not None and self.plan_anchor is not None
        age_frames = frame - self.plan_anchor
        lifted = self.model.lifted_plan(self.plan, age_frames // self.config.planbind_fine_spacing)
        residual = self.model.bind(lifted, age_frames, global_cond)
        return lifted + residual, residual.abs().amax(dim=(1, 2))

    @torch.no_grad()
    def step(self, global_cond_fn) -> Tensor:
        """Action ``(B, D)`` for the current frame; `global_cond_fn()` encodes the latest observations."""
        cfg = self.config
        spacing = cfg.planbind_fine_spacing
        latency = cfg.planbind_latency
        if self.pending is not None and self.frame >= self.pending[0]:
            self.cmd, self.cmd_anchor = self.pending[1], self.pending_anchor
            self.pending = None
        if (
            self.pending is None
            and self.frame % cfg.call_period == 0
            and (self.cmd is None or self.frame > self.cmd_anchor)
        ):
            chunk = self._call(global_cond_fn())
            if self.cmd is None:
                # Warm start: nothing is running yet, so the first chunk is used at once.
                frozen, swap = 0, self.frame
            else:
                frozen = min(n_frozen(latency, spacing, cfg.planbind_commit), chunk.shape[1])
                swap = self.frame + latency
                spans = torch.full((chunk.shape[0],), (self.frame - self.cmd_anchor) // spacing)
                running = shift_ctrl(self.cmd, spans)
                chunk = torch.cat([running[:, :frozen], chunk[:, frozen:]], dim=1)
            chunk = project(chunk, frozen, spacing, cfg.planbind_position_limit, cfg.planbind_velocity_limit)
            if swap <= self.frame:
                self.cmd, self.cmd_anchor = chunk, self.frame
            else:
                self.pending, self.pending_anchor = (swap, chunk), self.frame
        assert self.cmd is not None
        t = self.frame - self.cmd_anchor
        action = torch.einsum("n,bnd->bd", self.model.fine_basis[min(t, cfg.bspline_window - 1)], self.cmd)
        self.frame += 1
        return action
