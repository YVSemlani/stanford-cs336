import torch
import torch.nn as nn

import math

from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # register buffer for trig values
        self.register_buffer("trig_values", torch.zeros(max_seq_len, d_k), persistent=False)

        # create trig values given theta, d_k, and max_seq_len
        # we end up with d_k * max_seq_len ** 2 * 2 values (2 b/c sin and cos with negatives being an extension of these)

        for i in range(max_seq_len):
            for k in range(0, d_k, 2): # we attend to 2 dimensions at a time so we don't need if k % 2 == 0
                exponent = 2 * (k // 2) / d_k # k in the formula ranges from 0 to d_k / 2 thus representing a dimension pair rather than a single dimension
                self.trig_values[i, k] = math.cos(i / (theta ** exponent))
                self.trig_values[i, k + 1] = math.sin(i / (theta ** exponent))

    def forward(self, x, token_positions):

        # handle arbitrary batch dimensions for x and token_positions

        # if token positions and/or x don't have a batch dimension, add one
        #if token_positions.ndim == 1:
        #    token_positions = token_positions.view(1, -1)

        #if x.ndim == 2:
        #    x = x.view(1, -1, x.shape[-1])

        # for x and token positions, compress all batch dimensions into a single batch dimension

        x_copy = x.clone()
        token_positions_copy = token_positions.clone()

        x_copy = rearrange(x_copy, "... seq_len d_k -> (...) seq_len d_k")
        token_positions_copy = rearrange(token_positions_copy, "... seq_len -> (...) seq_len")

        # get our trig values using the token positions (shape: (A, seq_len, d_k))
        trig_values = self.trig_values[token_positions_copy]

        # get cosine and sine values for each dimension from 0 to the feature dim of x
        cos_values = trig_values[:, :, 0:x_copy.shape[-1]:2]
        sin_values = trig_values[:, :, 1:x_copy.shape[-1]:2]

        # get our odd dimensions (this breaks for certain combinations of num_heads and d_model which would lead to d_k being odd. i think we can leave that out for now)
        X_even = x_copy[:, :, 0::2]
        X_odd = x_copy[:, :, 1::2]

        # now we create the even and odd dimension rows
        X_even_rows = X_even * cos_values - X_odd * sin_values
        X_odd_rows = X_even * sin_values + X_odd * cos_values

        # now we can reassemble our X by interleaving the even and odd dimension rows
        output = torch.zeros_like(x_copy)
        output[:, :, 0::2] = X_even_rows
        output[:, :, 1::2] = X_odd_rows

        # now we can reshape our output to the same shape as x
        output = output.view(x.shape)

        return output


if __name__ == '__main__':
    x = torch.randn(1, 12, 12)
    token_positions = torch.arange(12)
    output = RoPE(12, 12, 512)(x, token_positions)
    print(output.shape)

