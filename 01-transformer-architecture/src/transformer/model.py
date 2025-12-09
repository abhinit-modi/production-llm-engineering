import torch.nn as nn

from transformer.blocks import TransformerBlock
from components.rope import get_rotation_matrix


class Transformer(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            hidden_size, 
            num_heads, 
            window_size, 
            d_ff, 
            num_experts, 
            n_experts_per_token, 
            n_blocks,
            max_seq_len
        ):

        super().__init__()

        head_dim = hidden_size // num_heads
        period = 10000.0
        self.rotation_matrix = get_rotation_matrix(head_dim, max_seq_len, period)

        # Initialize embedding, transformer blocks, and output projection
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.blocks = [TransformerBlock(hidden_size, num_heads, window_size, d_ff, num_experts, n_experts_per_token, self.rotation_matrix) for _ in range(n_blocks)]
        self.out = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, x):
        # Forward pass: embed -> transformer blocks -> project to vocabulary
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x