# Transformer Architecture from Scratch

Building modern LLM components from scratch using PyTorch — implementing the core building blocks found in state-of-the-art models like LLaMA, Mistral, and GPT.

## Overview

This project implements key transformer architecture components that power modern large language models:

- **Sliding Window Attention** — Memory-efficient attention with O(n·w) complexity instead of O(n²)
- **Rotary Position Embeddings (RoPE)** — Position encoding that enables better length generalization
- **SiGLU Activation** — Gated linear units for improved FFN performance
- **Mixture of Experts (MoE)** — Sparse expert routing with top-k gating
- **RMSNorm** — Efficient layer normalization used in LLaMA

## Architecture Components

### Sliding Window Attention

Many papers claim the Transformer architecture has quadratic time complexity O(N²) in the attention layer. While true for the original 2017 "Attention Is All You Need" paper, modern architectures use sparse attention mechanisms:

- **GPT-3** uses strided sparse attention
- **LLaMA 2** uses multihead attention with constant space complexity
- **Mistral 7B** uses sliding window attention

The implementation includes both a naive `SlidingWindowMultiheadAttention` and an optimized `EfficientSlidingWindowMultiheadAttention` using `torch.unfold` for vectorized operations.

**Complexity reduction:**
- Space: O(batch_size × num_heads × seq_len²) → O(batch_size × num_heads × seq_len × window_size)
- Time: O(batch_size × d_model × seq_len²) → O(batch_size × d_model × seq_len × window_size)

### Rotary Position Embeddings (RoPE)

Instead of adding positional encoding to semantic embeddings, RoPE rotates queries and keys based on token positions:

```
V' = R_θ · V
```

where R_θ is a rotation matrix. This embeds positional information directly into attention scores.

**Key advantages:**
- Relative position encoding through rotation
- Better extrapolation to longer sequences
- No additional parameters needed

### Mixture of Experts (MoE)

Sparse MoE layers route each token to a subset of "expert" networks:

1. **Gating**: Linear layer produces scores for each expert
2. **Top-k Selection**: Select k experts with highest scores
3. **Weighted Combination**: Softmax-normalize selected expert weights
4. **Expert Forward**: Route tokens through selected experts only

This enables scaling model capacity without proportionally increasing compute.

### SiGLU Activation

Gated Linear Unit with Sigmoid activation:

```
SiGLU(x) = W·x × σ(W_g·x)
```

Used in the FFN layers of modern transformers for improved training dynamics.

### RMSNorm

Root Mean Square Layer Normalization — a simplified and faster alternative to LayerNorm:

```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

## Project Structure

```
src/
├── components/
│   ├── activations.py    # SiGLU implementation
│   ├── attentions.py     # Sliding window attention (naive + efficient)
│   ├── moe.py            # Mixture of Experts layer
│   ├── norm_layers.py    # RMSNorm implementation
│   └── rope.py           # Rotary Position Embeddings
├── transformer/
│   ├── blocks.py         # Transformer block combining all components
│   └── model.py          # Full transformer model
└── test.ipynb            # Testing and validation
```

## Key Implementation Details

### Einstein Summation (einsum)

The efficient attention implementation uses `torch.einsum` for complex tensor operations:

```python
# Attention scores: Q·K^T for windowed attention
scores = torch.einsum('bnsh,bnshw->bnsw', queries, keys_windowed)

# Context: attention × values
context = torch.einsum('bnsw,bnshw->bsnh', attention, values_windowed)
```

### Complex Number Rotation for RoPE

RoPE uses complex number multiplication for efficient rotation:

```python
# Convert to complex representation
queries_complex = torch.view_as_complex(queries)
# Rotate via complex multiplication
queries_rotated = rotation_matrix * queries_complex
# Convert back to real
new_queries = torch.view_as_real(queries_rotated)
```

## Usage

```python
from transformer.model import Transformer

model = Transformer(
    vocabulary_size=50000,
    hidden_size=768,
    num_heads=12,
    window_size=256,
    d_ff=3072,
    num_experts=8,
    n_experts_per_token=2,
    n_blocks=12,
    max_seq_len=2048
)

output = model(input_ids)
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SiGLU
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Sliding Window Attention
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Mixture of Experts

