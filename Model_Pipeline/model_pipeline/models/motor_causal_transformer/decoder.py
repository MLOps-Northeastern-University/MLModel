"""The decoder: each window reads the windows before it, up to 8 hours back.

A decoder-only transformer over the window tokens (GPT-style, Radford et al. 2018), with
``layers`` pre-norm layers (Xiong et al. 2020, "On Layer Normalization in the Transformer"):

    x = x + Dropout(Out(Attention(LayerNorm(x))))        queries and keys rotated by real time
    x = x + Dropout(FeedForward(LayerNorm(x)))            Linear -> GELU -> Linear, 4 x wider
    ...
    output = LayerNorm(x)

Who sees whom. A window (query) attends to a token (key) only if all three hold:

    1. the key is not after it in the sequence             causal: never the future
    2. the key is less than history_s older                8 h by default, in real time
    3. the key is a real window, not padding

Rule 2 is on real time, not on token count: a stretch the watch wasn't worn counts as the time
that passed. RoPE on real time (rope.py) tells attention how far back each key is.

Reach. With 2 layers a window's output depends on up to 2 x history_s of the past, through its
keys' own pasts (``reach_s``). A window scored with less past than that sees less than it would
live; training and validation decide how much past each window gets.

How it runs fast:

- Everything that depends only on the tokens' times is worked out once per batch, in
  ``Decoder.reach``, and shared by every layer: the RoPE rotations and, for each block of
  queries, which keys it may see. The prototype rebuilt them in each layer.
- Queries are taken a block at a time (``query_block`` tokens, 1 h by default), each against
  only the keys its block can reach, so a day of tokens fits in memory. The output is the same
  as all queries at once. Finding each block's keys and checking the inputs costs one
  GPU-to-CPU sync per batch in all; the prototype synced once per block per layer.
- Queries and keys come out of one fused projection and are rotated in one call.
- Attention is PyTorch's ``scaled_dot_product_attention``, which picks a fused kernel for the
  device and dtype. On a GPU that is memory-efficient attention: FlashAttention takes no custom
  mask, and ours has the 8 h limit and padding. The plain fallback is about 3x slower.

Measured on an RTX 4060 Laptop GPU (batch 4 x 3,240 tokens, bf16, forward and backward): 22 ms
a training step against the prototype's 28 ms; a whole day scored (8,640 tokens) in 12 ms
against 23 ms. About 70% of the step is the attention kernel itself. Blocks of 1 h were the
fastest: 30 min blocks pay more launches, and one block for everything needs 2.3x the memory
and is slower. FlexAttention could skip the masked-out parts, but needs Triton (not on Windows).

Padding goes after the real tokens. A padding query is allowed to see only itself, so no row of
attention is empty (an empty softmax gives NaN, and 0 x NaN would then leak into real rows).
Its output is meaningless and ignored downstream.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.models.motor_causal_transformer.init import init_weights
from model_pipeline.models.motor_causal_transformer.rope import Rotation, TimeRoPE

HOUR_S = 3600.0


@dataclass(frozen=True)
class QueryBlock:
    """One block of queries and the keys it may see.

    Attributes:
        start, end: The block's queries are tokens start .. end - 1.
        first_key: The earliest key any query in the block can reach; keys are first_key .. end - 1.
        allowed: (batch, 1, end - start, end - first_key), True where that query may see that key.
    """

    start: int
    end: int
    first_key: int
    allowed: torch.Tensor


@dataclass(frozen=True)
class Reach:
    """What every layer shares for one batch: the rotations and who may see whom."""

    rotation: Rotation
    blocks: Tuple[QueryBlock, ...]


def _check_positive(name, value, integer=False):
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"decoder {name} must be a positive {'integer' if integer else 'number'}, not {value!r}")


class DecoderLayer(nn.Module):
    """One pre-norm transformer layer: causal attention on real time, then a feed-forward.

    Args:
        d_model: Numbers per token.
        heads: Attention heads; d_model must split evenly into them.
        ff_mult: The feed-forward's hidden width, as a multiple of d_model.
        dropout: Dropout on the attention weights and on both updates to x.
    """

    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim, self.dropout_p = heads, d_model // heads, dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)          # queries, keys and values in one matrix multiply
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Linear(ff_mult * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, reach: Reach, rope: TimeRoPE) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        qk = rope.rotate(qkv[:, :, :2], reach.rotation)     # (b, t, 2, heads, head_dim): q and k in one call
        # Heads first, as attention wants them: (b, heads, t, head_dim). Views, no copies.
        q, k, v = qk[:, :, 0].transpose(1, 2), qk[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        p = self.dropout_p if self.training else 0.0
        parts = [F.scaled_dot_product_attention(q[:, :, blk.start:blk.end], k[:, :, blk.first_key:blk.end],
                                                v[:, :, blk.first_key:blk.end], attn_mask=blk.allowed, dropout_p=p)
                 for blk in reach.blocks]
        att = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
        x = x + self.dropout(self.out(att.transpose(1, 2).reshape(b, t, d)))
        return x + self.dropout(self.ff(self.norm2(x)))


class Decoder(nn.Module):
    """Tokens in, one output per token: each token's view of itself and its past history_s.

    Args:
        d_model: Numbers per token (64).
        heads: Attention heads (4); d_model / heads must be even, for RoPE.
        layers: Decoder layers (2).
        ff_mult: Feed-forward width as a multiple of d_model (4).
        dropout: Dropout on attention weights and updates (0.3 in the final recipe).
        history_s: How far back, in seconds, a query sees keys (8 h).
        query_block: Queries per block (360: 1 h of 10-s windows); only memory and speed depend on it.
        rope_base, step_s: RoPE's settings (rope.py).
    """

    def __init__(self, d_model: int = 64, heads: int = 4, layers: int = 2, ff_mult: int = 4,
                 dropout: float = 0.3, history_s: float = 8 * HOUR_S, query_block: int = 360,
                 rope_base: float = 10_000.0, step_s: float = 10.0):
        super().__init__()
        for name, value in (("d_model", d_model), ("heads", heads), ("layers", layers), ("ff_mult", ff_mult),
                            ("query_block", query_block)):
            _check_positive(name, value, integer=True)
        _check_positive("history_s", history_s)
        if d_model % heads:
            raise ConfigError(f"decoder d_model ({d_model}) must split evenly into {heads} heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ConfigError(f"decoder dropout must be at least 0 and below 1, not {dropout!r}")
        self.d_model, self.history_s, self.query_block = d_model, float(history_s), query_block
        self.rope = TimeRoPE(d_model // heads, base=rope_base, step_s=step_s)
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads, ff_mult, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.apply(init_weights)

    @property
    def reach_s(self) -> float:
        """How far back a token's output can depend on: history_s per layer."""
        return self.history_s * len(self.layers)

    @torch.no_grad()
    def reach(self, time_s: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Reach:
        """Who may see whom, and the rotations, for one batch. Shared by every layer.

        Args:
            time_s: (batch, seq) each token's time in seconds, increasing along each sequence.
            valid: (batch, seq) True for real tokens, False for padding, which must come after them.
                None: every token is real.

        Raises:
            ModelError: times that aren't (batch, seq), aren't finite or go backwards, or padding
                before a real token.
        """
        if time_s.dim() != 2 or time_s.shape[1] == 0:
            raise ModelError(f"time_s must be (batch, seq) with at least one token, not {tuple(time_s.shape)}")
        device = self.norm.weight.device
        time_s = time_s.to(device=device, dtype=torch.float64)
        valid = (torch.ones_like(time_s, dtype=torch.bool) if valid is None
                 else valid.to(device=device, dtype=torch.bool))
        if valid.shape != time_s.shape:
            raise ModelError(f"valid {tuple(valid.shape)} must match time_s {tuple(time_s.shape)}")
        b, t = time_s.shape

        # Padding takes its sequence's last real time, so times never decrease and every one is
        # finite (a NaN or infinite time would turn into NaN rotations). An all-padding row gets 0.
        n_real = valid.sum(dim=1, keepdim=True)
        last_real = time_s.gather(1, (n_real - 1).clamp(min=0))
        last_real = torch.where(n_real > 0, last_real, torch.zeros_like(last_real))
        filled = torch.where(valid, time_s, last_real)

        # The earliest key each block of queries can reach: the first key less than history_s older
        # than its first query (the earliest, as times increase), found by binary search, then the
        # earliest over the batch. A row whose block starts in its padding needs nothing before the
        # block (padding only sees itself), so it doesn't widen the block for the rest of the batch:
        # without this, one short row in a batch made late blocks attend to almost every key.
        starts = list(range(0, t, self.query_block))
        oldest = filled[:, starts] - self.history_s
        first_key = torch.searchsorted(filled.contiguous(), oldest.contiguous(), right=True)
        block_start = torch.tensor(starts, device=device, dtype=first_key.dtype).expand_as(first_key)
        first_key = torch.where(valid[:, starts], first_key, block_start).min(dim=0).values

        # Every check and every block bound in one GPU-to-CPU copy.
        problems = torch.stack((
            (valid[:, 1:] & ~valid[:, :-1]).any(),                      # padding before a real token
            (~torch.isfinite(time_s) & valid).any(),                    # a real token's time not finite
            ((filled[:, 1:] < filled[:, :-1]) & valid[:, 1:]).any(),    # time going backwards
        )).to(torch.int64)
        host = torch.cat((problems, first_key)).tolist()
        if host[0]:
            raise ModelError("padding must come after a sequence's real tokens")
        if host[1]:
            raise ModelError("every real token's time must be finite")
        if host[2]:
            raise ModelError("time_s must increase along each sequence")

        index = torch.arange(t, device=device)
        blocks = []
        for start, lo in zip(starts, host[3:]):
            end = min(start + self.query_block, t)
            q_idx, k_idx = index[start:end, None], index[None, lo:end]
            gap = filled[:, start:end, None] - filled[:, None, lo:end]               # (b, queries, keys), seconds
            allowed = (k_idx <= q_idx) & (gap < self.history_s) & valid[:, None, lo:end]
            allowed |= (k_idx == q_idx) & ~valid[:, start:end, None]                # padding sees only itself
            blocks.append(QueryBlock(start, end, lo, allowed.unsqueeze(1)))         # (b, 1, q, k): every head
        # Rotate in float32 (also under bf16 mixed precision, see rope.py), or float64 for a float64 model.
        work = torch.float64 if self.norm.weight.dtype == torch.float64 else torch.float32
        return Reach(self.rope.rotation(filled, work), tuple(blocks))

    def forward(self, tokens: torch.Tensor, time_s: torch.Tensor, valid: Optional[torch.Tensor] = None,
                reach: Optional[Reach] = None) -> torch.Tensor:
        """Run the decoder.

        Args:
            tokens: (batch, seq, d_model).
            time_s: (batch, seq) each token's time in seconds, increasing along each sequence.
            valid: (batch, seq) False for padding, after the real tokens. None: all real.
            reach: From ``reach(time_s, valid)``, to reuse it across calls on the same tokens
                (such as several forward passes over one batch). None: worked out here.

        Returns:
            (batch, seq, d_model): each token's output.
        """
        if tokens.dim() != 3 or tokens.shape[-1] != self.d_model or tokens.shape[:2] != time_s.shape:
            raise ModelError(f"tokens must be (batch, seq, {self.d_model}) and time_s (batch, seq), "
                             f"not {tuple(tokens.shape)} and {tuple(time_s.shape)}")
        if reach is None:
            reach = self.reach(time_s, valid)
        elif reach.rotation.time_shape != tokens.shape[:2]:
            raise ModelError(f"reach is for tokens of shape {tuple(reach.rotation.time_shape)}, "
                             f"not {tuple(tokens.shape[:2])}")
        x = tokens
        for layer in self.layers:
            x = layer(x, reach, self.rope)
        return self.norm(x)
