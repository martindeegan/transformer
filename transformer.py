import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dim_k: int, dim_v: int):
        super(ScaledDotProductAttention, self).__init__()

        self.dim_k = dim_k
        self.dim_v = dim_v

        self.softmax = torch.nn.Softmax(dim_k)


    def forward(self, Q, K, V):
        return self.softmax(Q @ K.T / (self.dim_k ** 0.5)) @ V


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, model_dim: int = 512, h: int = 8, dim_k: int = 64, dim_v: int = 64):
        super(MultiHeadAttention, self).__init__()

        self.h = h

        self.linear_layers = []
        self.attention_layers = []
        for _ in range(h):
            W_Q = torch.nn.Linear(model_dim, dim_k, bias=False)
            W_K = torch.nn.Linear(model_dim, dim_k, bias=False)
            W_V = torch.nn.Linear(model_dim, dim_v, bias=False)
            self.linear_layers.append((W_Q, W_K, W_V))
            self.attention_layers.append(ScaledDotProductAttention(dim_k=dim_k, dim_v=dim_v))

        self.W_O = torch.nn.Linear(h * dim_v, model_dim, bias=False)

    def forward(self, Q, K, V):
        heads = []
        for i in range(self.h):
            W_Q, W_K, W_V = self.linear_layers[i]
            attention_layer = self.attention_layers[i]
            heads.append(attention_layer(W_Q(Q), W_K(K), W_V(V)))
        concatenated_heads = torch.concat(heads)
        return self.W_O(concatenated_heads)


