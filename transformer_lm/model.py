"""Decoder-only Transformer language model — all components from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from transformer_lm.nn_utils import silu, softmax


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------


class Linear(nn.Module):
    """Linear layer: ``y = x @ W^T + b``.

    Initialize ``weight`` uniformly in ``[-1/sqrt(in_features), 1/sqrt(in_features)]``.
    Initialize ``bias`` to zero. When ``bias=False``, set ``self.bias = None``
    and skip the bias addition in ``forward`` (the transformer model uses
    ``bias=False`` for all projections, so this is the common case).
    """

    """
    in_features = dimension of input features (only for init)
    out_features = dimension of output features (only for init)
    """
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()

        # Initialize weight (swapped order because of transpose)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-1/math.sqrt(in_features), 1/math.sqrt(in_features))
        )

        # Add bias (if wanted)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose and multiple
        transposed_output = x @ self.weight.T

        # Add bias (if not none)
        if self.bias is not None:
            bias_output = transposed_output + self.bias
            return bias_output
        else:
            # Return non-biased output
            return transposed_output
        


class Embedding(nn.Module):
    """Embedding look-up table.

    Initialize the weight matrix with normal distribution ``N(0, 0.02)``.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()

        # Initialize lookup table using randn for normal distribution 
        # (dividing by 50 / multiplying by 0.02 shifts the standard deviation to 0.02)
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.02
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Use lookup table to return correct vector set
        return self.weight[indices]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable parameters)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # Save eps as a self variable
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Square matrix
        squared = x.square()

        # Get mean (in last dimension)
        average = squared.mean(dim=-1, keepdim=True)

        # Add epsilon
        epsiloned = average + self.eps

        # Take square root
        squirted = epsiloned.sqrt()

        # Divide x by qurited
        return x / squirted




# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — PROVIDED, not student-implemented
# ---------------------------------------------------------------------------


# --- PROVIDED: do not modify ---
class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Pre-computes sin/cos rotation matrices and applies them to queries
    and keys.  This module is **provided** — students do not need to
    implement it.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        # Pre-compute for max_seq_len positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, d_head/2)
        cos_cached = torch.cos(freqs)  # (T, d_head/2)
        sin_cached = torch.sin(freqs)  # (T, d_head/2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: ``(B, n_heads, T, d_head)``
            k: ``(B, n_heads, T, d_head)``

        Returns:
            Rotated ``(q, k)`` with the same shapes.
        """
        T = q.shape[2]
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head/2)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        # Duplicate cos/sin to full d_head: [cos, cos] for pairs
        cos = torch.cat([cos, cos], dim=-1)  # (1, 1, T, d_head)
        sin = torch.cat([sin, sin], dim=-1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        Q: ``(B, ..., T, d_k)``
        K: ``(B, ..., T, d_k)``
        V: ``(B, ..., T, d_v)``
        mask: Additive mask. Use ``0`` for positions to attend to and a
            large negative value (e.g. ``-1e9``, also written as ``-inf``
            in some references) for positions to mask out.

    Returns:
        ``(B, ..., T, d_v)``
    """
    # First transpose K and multiply with Q (using @ for matrix multiplication)
    # We need to do the weird transpose of K because we want to do the dot product between the last dimension of Q and K, 
    # and the last two dimensions of K are T and d_k, 
    # so we need to swap them to get the correct shape for matrix multiplication.
    QK_product = Q @ K.transpose(-2, -1)

    # Divide by sqrt(d_k)
    d_k = Q.shape[-1]
    scaled_QK = QK_product / math.sqrt(d_k)

    # Apply softmax (written by me)
    if mask is not None:
        normalized = softmax(scaled_QK + mask)
    else:
        normalized = softmax(scaled_QK)

    # Multiply with V (suing @, not *)
    return normalized @ V


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask and RoPE.

    Uses a fused QKV projection for efficiency.

    Required attributes:
        qkv_proj: ``Linear(d_model, 3 * d_model, bias=False)`` — fused Q, K, V.
            Weight rows are packed as ``[Q_weights, K_weights, V_weights]``.
            Split with ``qkv.split(d_model, dim=-1)`` to recover Q, K, V.
        o_proj: ``Linear(d_model, d_model, bias=False)`` — output projection.
        rope: ``RotaryPositionEmbedding(d_head)`` — the default
            ``max_seq_len=4096`` is fine for our context length of 256.

    Args:
        d_model: Model dimension (must be divisible by ``n_heads``).
        n_heads: Number of attention heads (``d_model // n_heads`` must be even).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Assertions
        assert d_model % n_heads == 0
        assert (d_model // n_heads) % 2 == 0

        # Retain useful variables
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Set selfs
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        self.o_proj   = Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(d_model // n_heads)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``
        Returns:
            ``(B, T, d_model)``
        """
        # Get B and T
        B = x.shape[0]
        T = x.shape[1]

        # Apply fused projection: (B, T, 3*d_model)
        qkv = self.qkv_proj(x)   
        
        # Generate Q, K, and V: (B, T, d_model)
        Q, K, V = (qkv).split(self.d_model, dim=-1)

        # Reshape Q, K, V (have to use view in the wrong direction then correct)
        Q = Q.view(B, T, self.n_heads, (self.d_model // self.n_heads))
        Q = Q.transpose(1, 2)
        K = K.view(B, T, self.n_heads, (self.d_model // self.n_heads))
        K = K.transpose(1, 2)
        V = V.view(B, T, self.n_heads, (self.d_model // self.n_heads))
        V = V.transpose(1, 2)

        # Apply RoPE to Q and K
        Q, K = self.rope(Q, K)

        # Build an additive causal mask (−10^9 above the diagonal)
        mask = torch.triu(torch.ones(T, T), diagonal=1)
        mask = mask * -1e9

        # Compute scaled dot-product attention
        sdpa = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads back to (𝐵,𝑇,𝑑model) (reverse the annoying transpose and view from earlier)
        concatonate = sdpa.transpose(1, 2).reshape(B, T, self.d_model)

        # Apply the output projection, then dropout before the residual add in the block
        out = self.o_proj(concatonate)

        return self.dropout(out)



# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SwiGLU activation.

    ``FFN(x) = w_down(silu(w_gate(x)) * w_up(x))``

    Required attributes:
        w_gate: ``Linear(d_model, d_ff, bias=False)``
        w_up: ``Linear(d_model, d_ff, bias=False)``
        w_down: ``Linear(d_ff, d_model, bias=False)``

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        # Create required attributes
        self.w_gate = Linear(d_model, d_ff, bias=False)
        self.w_up = Linear(d_model, d_ff, bias=False)
        self.w_down = Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ####### Pass through all steps in feed forward
        ## Apply silu to w_gate(x)
        w_gate_x = self.w_gate(x)
        silu_output = silu(w_gate_x)

        ## Get w_up value
        w_up_x = self.w_up(x)

        ## Do element wise multiplication (⊙)
        element_wise = silu_output * w_up_x

        ## Feed into w_down
        w_down_x = self.w_down(element_wise)

        ## Apply dropout
        return self.dropout(w_down_x)

# ---------------------------------------------------------------------------
# Transformer block and full LM
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single pre-LN transformer decoder block.

    ``x = x + attn(ln1(x))``
    ``x = x + ffn(ln2(x))``

    Required attributes:
        ln1, ln2: ``RMSNorm(d_model)``
        attn: ``CausalMultiHeadSelfAttention(d_model, n_heads)``
        ffn: ``FeedForward(d_model, d_ff)``

    Note: zero-init of ``o_proj`` and ``w_down`` is done in
    ``TransformerLM.__init__``, **not** here.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        ###### Add Required Attributes
        ## RMSNorm Layers
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        ## Attention and Feed forward layers
        self.attn = CausalMultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply to the xes (and oh oh oh they want me!!!!)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
        


class TransformerLM(nn.Module):
    """Decoder-only transformer language model.

    Required attributes:
        token_emb: ``Embedding(vocab_size, d_model)``
        blocks: ``nn.ModuleList`` of ``TransformerBlock``
        ln_final: ``RMSNorm(d_model)``
        lm_head: ``Linear(d_model, vocab_size, bias=False)``
        context_length: store ``context_length`` so ``generate()`` and the
            overlength assertion in ``forward`` can read it.

    Weight tying: ``self.lm_head.weight = self.token_emb.weight``
    (must be the **same** ``nn.Parameter`` object, set in ``__init__``).

    Zero-init: after creating blocks, zero ``o_proj.weight`` and
    ``w_down.weight`` in every block so blocks start as identity.

    ``forward`` must raise ``AssertionError`` when the input sequence is
    longer than ``context_length``.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        context_length: Maximum sequence length.
        d_model: Embedding / hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        d_ff: FFN hidden dimension per block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        ####### Its all coming together https://a.pinatafarm.com/500x277/4bd38b530e/its-all-coming-together.jpg
        ## Toke Embedding
        self.token_emb = Embedding(vocab_size, d_model)
        ## Blocks (using list comprehension to create the correct number of blocks)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        ## Final Layer Norm and LM head
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        ## Store context length
        self.context_length = context_length

        # Fix the zero-init error (as laid out in the pdf)
        for block in self.blocks:
            block.attn.o_proj.weight.data.zero_()
            block.ffn.w_down.weight.data.zero_()

        # Weight tying (as laid out in the pdf)
        self.lm_head.weight = self.token_emb.weight


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: ``(B, T)`` integer token IDs.
        Returns:
            ``(B, T, vocab_size)`` raw logits.
        """

        ######## Begin the TRANSFORMATION
        ## Assert correct length
        assert input_ids.shape[1] <= self.context_length

        ## Get the token embedding
        x = self.token_emb(input_ids)

        ## Pass into blocks
        for block in self.blocks:
            x = block(x)

        ## Pass into ln_final
        x = self.ln_final(x)

        ## Pass into lm_head
        return self.lm_head(x)
