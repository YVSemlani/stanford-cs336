import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # initialize the embedding matrix and wrap in nn.Parameter to denote that it is a trainable parameter
        # shape is num_embeddings x embedding_dim so that each row is for a single token
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # initialize the embedding matrix from normal w/ mean 0, std 2 / sqrt(num_embeddings + embedding_dim), and clamped to be +-3
        self.weight = nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

        # convert to device and dtype if specified
        if dtype is not None:
            self.weight = self.weight.to(dtype)
        if device is not None:
            self.weight = self.weight.to(device)

    def forward(self, x):
        # treat embedding like a lookup table b/c one hot multiplication is analogous to a lookup table
        return self.weight[x]
    
if __name__ == "__main__":
    embedding = Embedding(10, 5)
    print("Model's state_dict:")
    for param_tensor in embedding.state_dict():
        print(param_tensor, "\t", embedding.state_dict()[param_tensor].size())

    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {embedding(x).shape}')