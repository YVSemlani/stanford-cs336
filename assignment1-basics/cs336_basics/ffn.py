import torch
import torch.nn as nn

from einops import rearrange, einsum

from .linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super(FFN, self).__init__()

        self.d_model = d_model
        if d_ff is None:
            self.d_ff = (8/3) * d_model
            # round down to the nearest multiple of 64
            self.d_ff = int(self.d_ff // 64) * 64
        else:
            self.d_ff = d_ff

        # dims are swapped from the paper b/c the linear layer weights are in the format (out_features, in_features)
        # thus our input to the linear layer is swapped compared to the paper
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def forward(self, x):
        inner_silu = self.w1(x)  # -> (batch_size, num_samples, d_ff)
        silu_out = inner_silu * torch.sigmoid(inner_silu)  # -> (batch_size, num_samples, d_ff)
        inner_out = silu_out * self.w3(x)  # -> (batch_size, num_samples, d_ff)
        out = self.w2(inner_out)  # -> (batch_size, num_samples, d_model)

        return out
    
if __name__ == "__main__":
    ffn = FFN(10, 5)
    print("Model's state_dict:")
    for param_tensor in ffn.state_dict():
        print(param_tensor, "\t", ffn.state_dict()[param_tensor].size())

    x = torch.randn(10, 10)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {ffn(x).shape}')
