# Scalable Attention Mechanisms

Extending LLM context windows through efficient attention architectures — implementing LongLoRA-style sparse grouped attention with RoPE scaling.

## Overview

This project fine-tunes LLaMA 3-8B to extend its context window using:

- **RoPE Position Interpolation** — Extend context without retraining position embeddings
- **Sparse Grouped Attention** — Reduce attention complexity from O(n²) to O(n·g)
- **Shifted Group Patterns** — Enable information flow between attention groups
- **LoRA Fine-tuning** — Parameter-efficient adaptation of attention layers

## The Challenge

Extending context size is conceptually simple but induces:
- **Quadratic complexity** in standard attention
- **Memory constraints** for long sequences
- **Training instability** at new sequence lengths

## Solution: LongLoRA Strategy

The approach combines multiple techniques:

### 1. RoPE Scaling

Instead of retraining position embeddings, interpolate between positions:

```python
# Original: θ = p / P^(2k/dim)
# Scaled:   θ = (p / scaling_factor) / P^(2k/dim)

config.rope_scaling = {
    'type': 'linear', 
    'factor': scaling_factor  # e.g., 2x for doubling context
}
```

### 2. Sparse Grouped Attention

Reduce operations by computing attention within groups:

**Standard Attention:**
- Each query attends to ALL keys: O(seq_len²)

**Grouped Attention:**
- Divide sequence into groups
- Each query only attends to keys in its group: O(seq_len × group_size)

```
Sequence: [t1, t2, t3, t4, t5, t6, t7, t8]
Groups:   [  Group 1   ][  Group 2   ]
          t1→{t1-t4}    t5→{t5-t8}
```

### 3. Shifted Groups

Problem: Information doesn't flow between groups.

Solution: Shift half the attention heads by `group_size // 2`:

```
Normal heads:  [t1,t2,t3,t4] [t5,t6,t7,t8]
Shifted heads: [t3,t4,t5,t6] [t7,t8,t1,t2]
```

This creates overlapping receptive fields.

## Implementation Details

### SparseAttention Forward Pass

```python
def forward(self, hidden_states, ...):
    batch_size, seq_len, _ = hidden_states.size()
    group_size = int(seq_len * self.group_size_ratio)
    num_groups = seq_len // group_size
    
    # Project to Q, K, V
    queries = self.q_proj(hidden_states)
    keys = self.k_proj(hidden_states)
    values = self.v_proj(hidden_states)
    
    # Apply RoPE
    cos, sin = self.rotary_emb(values, seq_len)
    queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
    
    # Shift half the heads
    queries = self.shift(queries, ...)
    keys = self.shift(keys, ...)
    values = self.shift(values, ...)
    
    # Compute grouped attention
    attn = softmax(Q @ K^T / sqrt(d)) @ V
    
    # Reverse shift and project output
    return self.o_proj(output)
```

### Shift Operation

```python
def shift(self, qkv, batch_size, seq_len, group_size, num_heads, head_dim):
    # Shift second half of heads by -group_size // 2
    qkv[:, num_heads//2:, :, :] = qkv[:, num_heads//2:, :, :].roll(
        -group_size // 2, dims=2
    )
    
    # Reshape: [batch, seq, heads, dim] → [batch*num_groups, group_size, heads, dim]
    qkv = qkv.reshape(batch_size * num_groups, group_size, num_heads, head_dim)
    
    return qkv.permute(0, 2, 1, 3)
```

### Complexity Analysis

| Attention Type | Space Complexity | Time Complexity |
|---------------|------------------|-----------------|
| Standard | O(batch × heads × seq²) | O(batch × d × seq²) |
| Grouped | O(batch × heads × seq × group) | O(batch × d × seq × group) |

With `group_size_ratio = 0.25`: **4x reduction** in attention compute.

## Data Collator

Custom collator ensures sequences are divisible by group size:

```python
class DataCollatorForSupervisedDataset:
    def __call__(self, instances):
        max_length = max(len(ids) for ids in input_ids)
        group_size = ceil(max_length * self.group_size_ratio)
        target_length = ceil(max_length / group_size) * group_size
        
        # Pad to target_length
        input_ids = [self.pad_sequence(ids, target_length, pad_token_id) ...]
        labels = [self.pad_sequence(ids, target_length, IGNORE_INDEX) ...]
```

## Project Structure

```
├── requirements.txt
└── src/
    ├── attention.py              # SparseAttention implementation
    ├── model_loading.py          # Model prep: RoPE scaling, LoRA, tokenizer
    ├── data_loading.py           # Dataset loading
    ├── data_processing.py        # Tokenization + custom collator
    ├── training.py               # LoRATrainer with cosine scheduler
    ├── training_application.py   # Main entry point
    ├── README.md                 # Technical notes
    └── test.ipynb               # Testing notebook
```

## Usage

```python
from model_loading import ModelLoader
from training import LoRATrainer

# Load model with extended context
loader = ModelLoader(
    model_id='meta-llama/Meta-Llama-3-8B',
    scaling_factor=2,      # 2x context extension
    group_size_ratio=0.25  # 4 groups
)
model, tokenizer = loader.load_and_prepare_model()

# Train
trainer = LoRATrainer(model, tokenizer, data_collator, group_size_ratio=0.25)
trainer.train(tokenized_data)
```

## Model Preparation Pipeline

1. **Load Config** — Get model hyperparameters
2. **Expand Context** — Apply RoPE scaling
3. **Load Model** — With optional 4-bit quantization
4. **Modify Attention** — Replace with SparseAttention
5. **Load Tokenizer** — Set extended `model_max_length`
6. **Resize Tokenizer** — Add special tokens, init embeddings
7. **Add LoRA** — Target attention projections

```python
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
```

## Key Equations

### RoPE Rotation Matrix

$$R_\theta(p) = \begin{pmatrix} \cos(p\theta_1) & -\sin(p\theta_1) \\ \sin(p\theta_1) & \cos(p\theta_1) \end{pmatrix}$$

### Grouped Query Attention (GQA)

LLaMA 3 uses GQA where:
- `num_attention_heads` — Number of query heads
- `num_key_value_heads` — Number of KV heads (shared across queries)
- `num_key_value_groups` — Queries per KV head

Benefits: Reduced KV-cache memory, faster inference.

## References

- [LongLoRA: Efficient Fine-tuning of Long-Context LLMs](https://arxiv.org/abs/2309.12307)
- [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)

