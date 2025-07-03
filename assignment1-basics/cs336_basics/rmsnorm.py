import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

        # convert to device and dtype if specified
        if dtype is not None:
            self.weight = self.weight.to(dtype)
        if device is not None:
            self.weight = self.weight.to(device)

    def forward(self, x):
        # upcast to float32 to avoid overflow
        original_dtype = x.dtype
        x = x.to(torch.float32)

        # calculate the RMS norm of the input
        rms_norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # apply the RMSNorm formula
        x = self.weight * x / rms_norm

        # downcast to original dtype
        x = x.to(original_dtype)

        return x
    
if __name__ == "__main__":
    rmsnorm = RMSNorm(10)
    print(rmsnorm.weight)

    x = torch.randn(10, 10)
    print(rmsnorm(x))