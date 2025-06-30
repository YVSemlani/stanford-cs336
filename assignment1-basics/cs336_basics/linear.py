import torch
import torch.nn as nn

from einops import rearrange, einsum

import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features


        # initialize the weight matrix as out x in b/c eqn is y = Wx (shapes are out x in * in x # samples)
        # wrap in nn.Parameter to denote that it is a trainable parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # set std to 2 / sqrt(in_features + out_features)
        self.std = math.sqrt(2 / (in_features + out_features))

        # initialize the weight matrix from normal w/ mean 0, std 2 / sqrt(in_features + out_features), and clamped to be within 3sigma of 0
        self.weight = nn.init.trunc_normal_(self.weight, mean=0, std=self.std, a=-3 * self.std, b=3 * self.std)

        # no bias based on assignment instructions

        # convert to device and dtype if specified
        if dtype is not None:
            self.weight = self.weight.to(dtype)
        if device is not None:
            self.weight = self.weight.to(device)

    def forward(self, x):
        return einsum(self.weight, x, "out_features in_features, batch_size num_samples in_features -> batch_size num_samples out_features")
    
if __name__ == "__main__":
    linear = Linear(10, 5)
    print("Model's state_dict:")
    for param_tensor in linear.state_dict():
        print(param_tensor, "\t", linear.state_dict()[param_tensor].size())
    
    # Test with different input shapes
    x1 = torch.randn(3, 10)  # batch_size=3, in_features=10
    x2 = torch.randn(2, 7, 10)  # batch_size=2, seq_len=7, in_features=10
    
    print(f"Input x1 shape: {x1.shape}")
    print(f"Output shape: {linear(x1).shape}")
    print(f"Input x2 shape: {x2.shape}")
    print(f"Output shape: {linear(x2).shape}")
            
