from components.rope import RoPE

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientSlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, rotation_matrix):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

        # Initialize RoPE for rotary position embeddings
        self.rope = RoPE(rotation_matrix)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Project input to queries, keys, and values
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Apply rotary position embeddings to queries and keys
        queries, keys = self.rope(queries, keys)

        # Pad keys and values for sliding window boundaries
        keys = F.pad(keys, (0, 0, padding, padding), "constant", 0)
        values = F.pad(values, (0, 0, padding, padding), "constant", 0)

        # Create sliding windows for keys and values using unfold
        # q, k and v are (batch_size, num_heads, seq_length, head_dim)
        # k_w and v_w are (batch_size, num_heads, num_windows, head_dim, window_size)
        # We select upto seq_len windows because, seq_len queries can attend to seq_len values ONLY
        keys_windowed = keys.unfold(2, self.window_size, 1)[:, :, :seq_length, :, :]
        values_windowed = values.unfold(2, self.window_size, 1)[:, :, :seq_length, :, :]
        
        
        # Compute scaled dot-product attention scores with einsum
        # q is of size (batch_size, num_heads, seq_length, head_dim)
        # k_w is (batch_size, num_heads, seq_len, head_dim, window_size)
        # scores is (batch_size, num_heads, seq_len, window_size)
        scores = torch.einsum('bnsh,bnshw->bnsw', queries, keys_windowed)
        scores = scores / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        # Apply attention weights to values using einsum
        # values_windows (batch_size, num_heads, seq_len, head_dim, window_size)
        # attention is (batch_size, num_heads, seq_len, window_size)
        # context is (batch_size, seq_len, num_heads, head_dim)
        context = torch.einsum('bnsw,bnshw->bsnh', attention, values_windowed)

        # Merge attention heads back to hidden dimension
        context = context.reshape(batch_size, seq_length, self.hidden_size)

        # Final output projection
        output = self.out(context)
        return output
   
    
class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Pad sequence for windowed attention
        keys = F.pad(keys, (0, 0, padding, padding), "constant", 0)
        values = F.pad(values, (0, 0, padding, padding), "constant", 0)

        # Initialize context tensors
        context = torch.zeros_like(queries, device=x.device)

        # Compute attention for each sliding window
        for i in range(seq_length):
            # Determine the start and end of the window
            start = i
            end = i + self.window_size
            
            # Compute scores
            # (batch_size, num_heads, seq_length, head_dim)
            scores = torch.matmul(queries[:, :, i:i+1, :], keys[:, :, start:end, :].transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention to values and add to context
            context[:, :, i:i+1, :] += torch.matmul(attention, values[:, :, start:end, :])

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)

        # Final linear layer
        output = self.out(context)
        return output