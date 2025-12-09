import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # How is nn.Parameter different than nn.Linear? Is it trainable?
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        # TODO: implement the normalization
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight