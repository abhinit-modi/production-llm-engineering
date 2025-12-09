import torch
import torch.nn as nn


class SiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W_g = nn.Linear(in_features, out_features, bias=False)
  
    def forward(self, x):
        # TODO: implement SiGLU W * x * sigma (W_g * x)
        return self.W(x) * torch.sigmoid(self.W_g(x))