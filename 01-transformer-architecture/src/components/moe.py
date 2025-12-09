from components.activations import SiGLU

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        # TODO: instantiate 3 linear layers
        self.W1 = nn.Linear(hidden_size, d_ff)
        self.W2 = nn.Linear(hidden_size, d_ff)
        self.W3 = nn.Linear(d_ff, hidden_size)
        self.siglu = SiGLU(d_ff, d_ff)

    def forward(self, x) -> torch.Tensor:
        # TODO: implement the expert logic
        # x is of shape batch_size, seq_len, hidden_size
        # output is batch_size, seq_len, hidden_size
        x1 = self.W1(x)
        x2 = self.W2(x)
        out = self.siglu(x1*x2)
        out = self.W3(out)
        return out


class MoeLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super().__init__()

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        # TODO: instantiate the experts and the gate
        self.gate = nn.Linear(hidden_size, num_experts) 
        self.experts = [FeedForward(hidden_size, d_ff) for _ in range(num_experts)] 

    def forward(self, x):
        # TODO: pass the input x to the gate
        # x is of shape (batch_size, seq_len, hidden_size)
        # After gating it is (batch_size, seq_len, num_experts) i.e for each token in seq there are num_expert scores
        y = self.gate(x)

        # TODO: use torch.topk to get the topk values and indexes
        # topk has shape batch_size, seq_len, n_experts_per_tokenx
        # topk output is a tuple of (values, indices)
        # values are the scores for the topk experts, indices are the expert indices
        topk = torch.topk(y, self.n_experts_per_token, -1)

        # TODO: pass the topk values to F.softmax to get the weights for each expert 
        # topk_weights has shape batch_size, seq_len, n_experts_per_tokenx
        topk_weights = F.softmax(topk.values, dim=-1)
  
        out = torch.zeros_like(x, device=x.device)
        for i, expert in enumerate(self.experts):
            # topk.indices is of shape (batch_size, seq_len, n_experts_per_token)
            # torch.where returns the indices of the topk experts for each token
            # batch_idx, token_idx, topk_idx are the **indices** of the batches, tokens, and chosen_experts
            batch_idx, token_idx, topk_idx = torch.where(topk.indices == i)
            out[batch_idx, token_idx] += topk_weights[batch_idx, token_idx, topk_idx, None] * expert(x[batch_idx, token_idx])

        return out