
# Making Architectureal Change to the Base Model Structure

## RoPE changes

* In RoPE, embeddings are rotated in parts. Each token's embedding is broken down into $dim/2$ pairs of (x, y) coordinates i.e lines. Each of them is then rotated by different frequencies. The ones at the beginning index of the embedding are rotated by a larger frequency to capture small context dependencies, the ones towards the end are rotated slowly, to capture long range contexts. We can reduce (Scaling factor) the frequency to capture longer dependencies. 

* But to capture longer context, we want to deal with more keys and queries. We would need to optimize attention computation

## Adding tokens to vocabulary

* tokenizer.add_special_tokens - use this to insert the tokens into the tokenizer
* model.resize_token_embeddings - use this to let the model know about the new tokens added
* model.set_[input|output]_embedding to assign values to the new tokens


## Attention changes

* In standard attention, each head looks at a part of the hidden_state (d_model vs head_dim), but it looks at the entire sequence seq_len.
* In sliding window attention, in each head the search space (keys) is reduced to window_size, query space is still seq_len
* In grouped attention, in each head both search and key_space are reduced to group_size
    * the information does not flow between groups. To counter this, they rotated the 

### Sliding window
Q = batch_size * num_heads * seq_len * d_head
K = batch_size * num_heads * seq_len * d_head * window_size
Q.Kt = batch_size * num_heads * seq_len * window_size (without windows this would be seq_len * seq_len)

* Sliding window attention reduces the space complexity of self attention from $O(batch_size*num_heads*seq_len^2)$ to $O(batch_size*num_heads*seq_len*window_size)$
* It also changes the time complexity from $O(batch_size*d_model*seq_len^2)$ to $O(batch_size*d_model*seq_len*window_size)$

### Sliding window grouped attention
Q = batch_size * num_heads * num_groups * window_size * d_head
K = batch_size * num_heads * num_groups * window_size * d_head
Q.Kt = batch_size * num_heads * num_groups * window_size * window_size (without groups this would be seq_len * window_size)

* Sliding window attention reduces the space complexity of self attention from $O(batch_size*num_heads*seq_len^2)$ to $O(batch_size*num_heads*seq_len*window_size)$
* It also changes the time complexity from $O(batch_size*d_model*seq_len^2)$ to $O(batch_size*d_model*seq_len*window_size)$



