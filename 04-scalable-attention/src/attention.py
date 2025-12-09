from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math

class SparseAttention(LlamaAttention):

    def __init__(self, config: LlamaConfig, group_size_ratio=0.25, layer_index=None):
        super().__init__(config, layer_idx=layer_index)
        self.group_size_ratio = group_size_ratio

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # get the dimensions
        batch_size, seq_len, hidden_size = hidden_states.size()
        # Calculate group size based on sequence length and ratio
        group_size = int(seq_len * self.group_size_ratio)
        # Compute number of groups for grouped attention
        num_group = seq_len // group_size

        # Project hidden states to queries, keys, and values using Llama's projections
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape to separate heads: [batch_size, seq_len, num_heads, head_dim]
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim] for attention
        queries = queries.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        keys = keys.permute(0, 2, 1, 3)        # [batch_size, num_heads, seq_len, head_dim]
        values = values.permute(0, 2, 1, 3)    # [batch_size, num_heads, seq_len, head_dim]

        # Apply rotary position embeddings to queries and keys
        cos, sin = self.rotary_embed(values, seq_len)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids)

        # Repeat KV heads for grouped-query attention compatibility
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        # Apply shift operation for sparse grouped attention
        queries = self.shift(queries, batch_size, seq_len, group_size, self.num_heads, self.head_dim)
        keys = self.shift(keys, batch_size, seq_len, group_size, self.num_heads, self.head_dim)
        values = self.shift(values, batch_size, seq_len, group_size, self.num_heads, self.head_dim)

        # Compute scaled dot-product attention scores
        interaction_matrix = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # compute local mask
        local_mask = self.shift_attention_masks(
            simple=True, 
            attention_mask=attention_mask, 
            group_size=group_size, 
            batch_size=batch_size, 
            num_group=num_group, 
            num_heads=self.num_heads
        )

        # Apply local attention mask and softmax normalization
        attn_weights = nn.functional.softmax(local_mask + interaction_matrix, dim=-1)
        
        # Compute context by applying attention weights to values
        new_hidden_states = torch.matmul(attn_weights, values)
        
        # Reshape from [batch_size * num_group, num_heads, group_size, head_dim]
        # back to [batch_size, seq_len, num_heads, head_dim]
        new_hidden_states = new_hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reverse the shift for the shifted heads (roll back by +group_size//2)
        new_hidden_states[:,:,self.num_heads//2:,:] = new_hidden_states[:,:,self.num_heads//2:,:].roll(group_size // 2, dims=1)

        # Merge heads and apply output projection
        new_hidden_states = new_hidden_states.reshape(batch_size, seq_len, self.hidden_size)
        new_hidden_states = self.o_proj(new_hidden_states)

        if not output_attentions:
            attn_weights = None

        return new_hidden_states, attn_weights, past_key_value
    
    def shift(self, qkv, batch_size, seq_len, group_size, num_heads, head_dim):      
        # Shift operation for sparse grouped attention:
        # 1. Roll half the heads by -group_size//2 to enable cross-group information flow
        # 2. Reshape to increase batch dimension while reducing sequence length to group_size
        # 3. Final shape: [batch_size * num_groups, num_heads, group_size, head_dim]

        qkv[:,num_heads//2:,:,:] = qkv[:,num_heads//2:,:,:].roll(-group_size // 2, dims=2)
        qkv = qkv.reshape(batch_size, seq_len, num_heads, head_dim)
        qkv = qkv.reshape(batch_size * (seq_len // group_size), group_size, num_heads, head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size * (seq_len / group_size), num_heads, group_size, head_dim]
        return qkv
    
    def shift_attention_masks(self, simple, attention_mask, group_size, batch_size, num_group, num_heads):
        
        if simple:
            # The way done by LongLoRA
            return attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)

        # shift the queries and the keys
        # [batch_size, 1, seq_len, seq_len]
        attention_mask_rolled = attention_mask.roll(-group_size // 2, dims=-1).roll(-group_size // 2, dims=-2)
        # reshape per group
        # [batch_size, 1, num_groups, group_size, num_groups, group_size]
        attention_mask = attention_mask.reshape(batch_size, 1, num_group, group_size, num_group, group_size)
        attention_mask_rolled = attention_mask_rolled.reshape(batch_size, 1, num_group, group_size, num_group, group_size)

        # repeat dim = 1 for half the number of heads
        attention_mask = attention_mask.expand(batch_size, num_heads // 2, num_group, group_size, num_group, group_size)
        attention_mask_rolled = attention_mask_rolled.expand(batch_size, num_heads // 2, num_group, group_size, num_group, group_size)
 
        # concatenate the non shifted attention masks with the shifted ones
        attention_mask = torch.cat([attention_mask, attention_mask_rolled], dim=1)

        # We want to keep only the masks for the groups on the diagonal
        # [num_group, 1, num_group, 1]
        diagonal_mask = torch.eye(num_group, dtype=torch.bool).unsqueeze(-2).unsqueeze(-1)
        # We repeat the pattern for all dimensions
        # [batch_size, num_heads, num_group, group_size, num_group, group_size]
        diagonal_mask = diagonal_mask.expand(batch_size, num_heads, num_group, group_size, num_group, group_size)

        # We keep only the masks for the groups on the diagonal
        local_attention_mask = attention_mask.masked_select(diagonal_mask)
        local_attention_mask = local_attention_mask.view(batch_size, num_heads, num_group, group_size, group_size)
        # We reshape to match the attention shape
        local_attention_mask = local_attention_mask.transpose(1, 2).reshape(batch_size * num_group, num_heads, group_size, group_size)

        return local_attention_mask