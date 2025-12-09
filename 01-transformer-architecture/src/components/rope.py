import torch
import torch.nn as nn
from typing import Tuple


def get_rotation_matrix(dim: int, context_size: int, period: float) -> torch.Tensor:
    # Compute frequency bands for each dimension pair
    exps = (1./dim) * torch.arange(0, (dim-1), 2)
    freqs = (1./torch.pow(period, exps))

    # Create position indices for the sequence
    token_indexes = torch.arange(0, context_size)
    
    # Compute rotation angles (theta) as outer product of positions and frequencies
    thetas = torch.outer(token_indexes, freqs)

    # Create complex rotation matrix using polar form (magnitude=1, angle=theta)
    rotation_matrix = torch.polar(torch.ones_like(thetas), thetas)

    return rotation_matrix


class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super().__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.size()

        # Reshape to pair adjacent dimensions for complex representation
        # [batch_size, num_heads, seq_length, head_dim // 2 , 2]
        queries = queries.reshape((batch_size, num_heads, seq_length, head_dim // 2 , 2))
        keys = keys.reshape((batch_size, num_heads, seq_length, head_dim // 2 , 2))

        # Convert to complex tensors for rotation
        queries_complex = torch.view_as_complex(queries)
        keys_complex = torch.view_as_complex(keys)

        # Apply rotation via complex multiplication
        queries_rotated = self.rotation_matrix[:seq_length, :]*queries_complex
        keys_rotated = self.rotation_matrix[:seq_length, :]*keys_complex

        # Convert back to real and reshape to original dimensions
        new_queries = torch.view_as_real(queries_rotated).reshape((batch_size, num_heads, seq_length, head_dim))
        new_keys = torch.view_as_real(keys_rotated).reshape((batch_size, num_heads, seq_length, head_dim))

        return new_queries, new_keys











def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)