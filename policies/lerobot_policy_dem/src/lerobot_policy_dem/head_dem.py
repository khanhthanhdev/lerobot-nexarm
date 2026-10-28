"""MeanFlow head whose AdaLN conditioning also carries the instruction and the state.

In the paper's head the observation reaches the action tokens only through gated cross-attention into
the context, where the instruction is 32 of several hundred tokens and the state is one. On NexArm
stack-bowls (three cameras, 801 context tokens) a head trained that way ignored both: the six stacking
orders, which only the prompt tells apart, gave the same chunk. This head adds a global vector to the
interval embedding `c` that modulates every block:

    g = MLP(LayerNorm([attention-pool_k(L W) for k in 1..K] || q W_q))
    c = r_emb(r) + t_emb(t) + g

Learned queries pool the language tokens (a plain mean is dominated by the words the prompts share),
and the z-scored state enters directly. The last layer of the MLP is zero-initialised, so at step 0 the
head computes exactly what `MeanFlowTokenHead` computes. The attention is `dem`'s explicit-matmul
implementation, which the MeanFlow Jacobian-vector product requires.
"""

import torch
from dem.head import MeanFlowTokenHead
from dem.head.layers import ManualCrossAttention
from torch import Tensor, nn


class GlobalCondMeanFlowHead(MeanFlowTokenHead):
    """`MeanFlowTokenHead` plus instruction and state conditioning through AdaLN."""

    def __init__(self, *args: object, num_language_queries: int = 4, **kwargs: object) -> None:
        """Build the paper's head, then the global conditioning branch."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        dim = self.dim
        self.global_lang_proj = nn.Linear(self.lang_dim, dim)
        self.global_lang_queries = nn.Parameter(torch.randn(num_language_queries, dim) * 0.02)
        self.global_lang_pool = ManualCrossAttention(dim, self.heads)
        self.global_prop_proj = nn.Linear(self.proprio_dim, dim)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm((num_language_queries + 1) * dim),
            nn.Linear((num_language_queries + 1) * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.global_mlp[-1].weight)
        nn.init.zeros_(self.global_mlp[-1].bias)

    def global_cond(self, cond: dict) -> Tensor:
        """Instruction and state vector `[B, dim]` added to the interval embedding."""
        lang = self.global_lang_proj(cond["lang_tokens"])
        batch = lang.shape[0]
        mask = cond.get("lang_mask")
        if mask is None:
            mask = torch.ones(lang.shape[:2], dtype=torch.bool, device=lang.device)
        queries = self.global_lang_queries.expand(batch, -1, -1)
        pooled = self.global_lang_pool(queries, lang, mask.bool()).flatten(1)
        prop = self.global_prop_proj(cond["proprio"])
        return self.global_mlp(torch.cat([pooled, prop], dim=-1))

    def forward(self, z: Tensor, r: Tensor, t: Tensor, cond: dict, return_hidden: bool = False):
        """`MeanFlowTokenHead.forward` with `global_cond` added to the AdaLN conditioning."""
        ctx, mask = cond["_ctx"] if "_ctx" in cond else self.build_ctx(cond)
        c = self.r_emb(r) + self.t_emb(t) + self.global_cond(cond)
        h = self.a_in(z) + self.pos
        mid = None
        for i, blk in enumerate(self.blocks):
            h = blk(h, c, ctx, mask)
            if return_hidden and i == len(self.blocks) // 2:
                mid = h
        u = self.out(self.out_norm(h))
        return (u, mid) if return_hidden else u
