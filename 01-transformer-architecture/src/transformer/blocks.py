import torch.nn as nn

from components.norm_layers import RMSNorm
from components.attentions import EfficientSlidingWindowMultiheadAttention
from components.moe import MoeLayer


class TransformerBlock(nn.Module):
    def __init__(
          self, 
          hidden_size, 
          num_heads, 
          window_size, 
          d_ff, 
          num_experts, 
          n_experts_per_token,
          rotation_matrix
        ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        self.layer_norm_1 = RMSNorm(self.hidden_size)
        self.attention_layer = EfficientSlidingWindowMultiheadAttention(self.hidden_size, self.num_heads, self.window_size, rotation_matrix)
        self.layer_norm_2 = RMSNorm(self.hidden_size)
        self.feed_forward = MoeLayer(self.hidden_size, self.d_ff, self.num_experts, self.n_experts_per_token)


    def forward(self, x):
        y = self.layer_norm_1(x)
        y = self.attention_layer(y)
        x = x + y
        y = self.layer_norm_2(x)
        y = self.feed_forward(y)
        return x + y

        




