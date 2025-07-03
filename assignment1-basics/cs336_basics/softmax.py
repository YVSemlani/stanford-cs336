import torch
import torch.nn as nn

from einops import einsum

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()



    def forward(self, x, dim):

        # get the maxes of the tensor along the dimension to apply softmax to
        maxes = x.max(dim=dim, keepdim=True).values

        # subtract the maxes from the tensor to avoid overflow
        x_adjusted = x - maxes

        # exponentiate the tensor
        x_adjusted = x_adjusted.exp()

        x_denominator = x_adjusted.sum(dim=dim, keepdim=True)

        # divide the exponentiated tensor by the sum of the remaining dimensions
        x_adjusted = x_adjusted / x_denominator

        return x_adjusted



if __name__ == "__main__":
    x = torch.randn(2, 2, 3, 4)
    dim = 2
    softmax = Softmax()
    print(softmax(x, dim))

    # pytorch softmax
    print(torch.softmax(x, dim=dim))