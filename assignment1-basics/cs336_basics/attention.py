import torch 
import torch.nn as nn

import math

from einops import einsum, rearrange

if __name__ == "__main__":
    from softmax import Softmax
    from linear import Linear
    from RoPE import RoPE
else:
    from .softmax import Softmax
    from .linear import Linear
    from .RoPE import RoPE

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.softmax = Softmax()

    def forward(self, Q, K, V, mask=None):
        # NOTE: we don't have to think about the shapes b/c einsum handles it :) i <3 einsum

        # get the dimensions of the input tensors
        batch_size, seq_len, d_k = Q.shape[0], Q.shape[-2], Q.shape[-1]
        d_v = V.shape[-1]

        # compute the pre softmax attention scores
        attention_scores = einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k") / math.sqrt(d_k)

        # apply the mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == False, float('-inf'))

        # apply softmax to the masked attention scores
        attention_scores = self.softmax(attention_scores, dim=-1)

        # apply the attention scores to the values
        attention_output = einsum(attention_scores, V, "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")

        # return the attention output
        return attention_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=1024, theta=10000, use_rope=False):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = theta
        self.use_rope = use_rope

        # set the dimensions of the attention heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k # we assume d_v = d_k by the instructions

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        self.output_proj = Linear(d_model, d_model)

        if self.use_rope:
            self.rope = RoPE(self.rope_theta, self.d_k, self.max_seq_len)

        self.attention_heads = [Attention() for _ in range(self.num_heads)]

    def forward(self, x, token_positions=None):
        
        # project the queries, keys, and values using the linear layers
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # generate the mask which will be the same shape as the inner attention scores (batch_size, seq_len_q, seq_len_k)
        # causal mask is implicit 

        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2]), diagonal=0) # we want the shape of the inner attention scores so we take seq_len_q and seq_len_k
        mask = mask.bool()

        # split the queries, keys, and values along the feature dimension into num_heads groups
        # starts as (batch_size, seq_len, d_model)
        # ends as (batch_size, num_heads, seq_len, d_model // num_heads)

        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.d_v).transpose(1, 2)

        # apply RoPE to the queries and keys
        # we apply RoPE post rearrangement so that batch size and num heads are both treated as batch dims
        # thus the same RoPE is applied to all heads in the batch

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(Q.shape[-2])
            else:
                token_positions = token_positions

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # apply the attention mechanism to each head
        # same mask for all heads so we can just pass it in once
        attention_outputs = [attention_head(Q[:, i, :, :], K[:, i, :, :], V[:, i, :, :], mask=mask) for i, attention_head in enumerate(self.attention_heads)]

        # concatenate the attention outputs along the feature dimension
        # each ele. of the list is (batch_size, seq_len, d_head) and there are num_heads of them so concat is (batch_size, seq_len, d_model)
        attention_outputs = torch.cat(attention_outputs, dim=-1)

        # project the attention outputs back to the original dimension
        return self.output_proj(attention_outputs)
    
if __name__ == "__main__":
    x = torch.randn(2, 10, 1024) # (batch_size, seq_len, d_embedding)
    attention = MultiHeadAttention(d_model=1024, num_heads=8, max_seq_len=1024, theta=10)
    print(attention(x).shape)