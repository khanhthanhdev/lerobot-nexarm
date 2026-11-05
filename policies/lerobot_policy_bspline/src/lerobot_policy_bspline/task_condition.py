"""Task-text conditioning for the B-spline policies (`task_condition="hash"`).

LIBERO suites hold several tasks per scene, so a policy without the instruction cannot succeed. The
instruction is hashed word by word into `task_hash_buckets` ids (``zlib.crc32``, stable across processes),
embedded and mean-pooled, and appended to the observation features that condition the UNet. No text
model is downloaded, and any instruction string works.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

TASK_TOKENS = "task_tokens"
TASK_EMBED_DIM = 128


def tokenize_tasks(tasks: Sequence[str], buckets: int, max_tokens: int) -> Tensor:
    """Hashed word ids ``(B, max_tokens)`` of each instruction; 0 is padding."""
    ids = torch.zeros(len(tasks), max_tokens, dtype=torch.long)
    for row, task in enumerate(tasks):
        words = task.lower().replace("_", " ").split()[:max_tokens]
        for col, word in enumerate(words):
            ids[row, col] = zlib.crc32(word.encode()) % (buckets - 1) + 1
    return ids


class TaskConditionMixin:
    """Put before `DiffusionModel` in the bases: appends the task embedding to the global condition."""

    def __init__(self, config):
        """Build the diffusion modules; with a task condition, rebuild the UNet for the wider condition."""
        super().__init__(config)
        if config.task_condition == "none":
            return
        cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            encoder = self.rgb_encoder[0] if config.use_separate_rgb_encoder_per_camera else self.rgb_encoder
            cond_dim += encoder.feature_dim * len(config.image_features)
        if config.env_state_feature:
            cond_dim += config.env_state_feature.shape[0]
        self.task_embed = nn.Embedding(config.task_hash_buckets, TASK_EMBED_DIM, padding_idx=0)
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=cond_dim * config.n_obs_steps + TASK_EMBED_DIM
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        cond = super()._prepare_global_conditioning(batch)
        if self.config.task_condition == "none":
            return cond
        tokens = batch[TASK_TOKENS].to(cond.device)
        mask = (tokens > 0).unsqueeze(-1).to(cond.dtype)
        pooled = (self.task_embed(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.cat([cond, pooled], dim=-1)
