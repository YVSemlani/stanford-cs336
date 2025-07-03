import torch 
import torch.nn as nn

if __name__ == '__main__':
    from attention import MultiHeadAttention
    from ffn import FFN
    from rmsnorm import RMSNorm
    from embedding import Embedding
    from linear import Linear
    from RoPE import RoPE
    from softmax import Softmax
else:
    from .attention import MultiHeadAttention
    from .ffn import FFN
    from .rmsnorm import RMSNorm
    from .embedding import Embedding
    from .linear import Linear
    from .RoPE import RoPE
    from .softmax import Softmax

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff=None, theta=None):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.d_model = d_model
        self.num_heads = num_heads
        if d_ff is None:
            self.d_ff = (8/3) * self.d_model
            self.d_ff = int(self.d_ff // 64) * 64
        else:
            self.d_ff = d_ff

        if theta is None:
            self.theta = 10000
        else:
            self.theta = theta

        self.token_embeddings = Embedding(self.vocab_size, self.d_model)

        self.layers = torch.nn.ModuleList([TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, self.theta) for _ in range(self.num_layers)])

        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)

    def forward(self, indices):
        # get the token embeddings
        x = self.token_embeddings(indices)
        
        # pass through the transformer layers
        for layer in self.layers:
            x = layer(x)

        # apply final layer norm and project to the vocab size
        x = self.ln_final(x)

        # project to vocab size
        x = self.lm_head(x)

        # apply softmax over the last dim (vocab size) to get the probabilities of each token in the vocab
        #x = torch.softmax(x, dim=-1)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.max_seq_len = max_seq_len
        self.theta = theta

        self.attn = MultiHeadAttention(self.d_model, self.num_heads, max_seq_len=self.max_seq_len, theta=self.theta, use_rope=True)
        self.ffn = FFN(self.d_model, d_ff=self.d_ff)
        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)

    def forward(self, x):
        x += self.attn(self.ln1(x)) # sublayer 1
        x += self.ffn(self.ln2(x)) # sublayer 2
        return x