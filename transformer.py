import torch
import math
from tqdm import tqdm

def default_params():
    return {
        "model_dim": 512,
        "h": 8,
        "dim_k": 64,
        "dim_v": 64,
        "dim_ff": 2048,
        "transformer_blocks": 2,
        "context_length": 8,
        "dropout": 0.1
    }


class SelfAttentionHead(torch.nn.Module):
    def __init__(self, params):
        super(SelfAttentionHead, self).__init__()
        self.dim_k = params["dim_k"]
        self.key = torch.nn.Linear(params["model_dim"], params["dim_k"])
        self.value = torch.nn.Linear(params["model_dim"], params["dim_v"])
        self.query = torch.nn.Linear(params["model_dim"], params["dim_k"])
        self.register_buffer("tril", torch.tril(torch.ones(params["context_length"], params["context_length"])))
        self.dropout = torch.nn.Dropout(params["dropout"])

    def forward(self, Q, K, V):
        B,T,C = Q.shape
        proj_Q = self.query(Q)
        proj_K = self.key(K)
        proj_V = self.value(V)

        weights = proj_Q @ proj_K.transpose(-2, -1) / math.sqrt(self.dim_k)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        x = weights @ proj_V
        return x

    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()

        self.h = params["h"]

        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(params)
            for _ in range(self.h)
        ])
        self.W_O = torch.nn.Linear(params["h"] * params["dim_v"], params["model_dim"], bias=False)
        self.dropout = torch.nn.Dropout(params["dropout"])

    def forward(self, Q, K, V):
        heads = [head(Q, K, V) for head in self.heads]
        concatenated_heads = torch.cat(heads, dim=-1)
        out = self.W_O(concatenated_heads)
        return self.dropout(out)


class FeedForward(torch.nn.Module):
    def __init__(self, params):
        super(FeedForward, self).__init__()

        model_dim = params["model_dim"]
        dim_ff = params["dim_ff"]

        self.feed_forward_1 = torch.nn.Linear(model_dim, dim_ff)
        self.relu = torch.nn.ReLU()
        self.feed_forward_2 = torch.nn.Linear(dim_ff, model_dim)
        self.dropout = torch.nn.Dropout(params["dropout"])

    def forward(self, x):
        x = self.feed_forward_1(x)
        x = self.relu(x)
        x = self.feed_forward_2(x)
        x = self.dropout(x)
        return x

class MaskedMultiHeadAttention(torch.nn.Module):
    def __init__(self, params):
        super(MaskedMultiHeadAttention, self).__init__()

        self.h = params["h"]
        
        model_dim = params["model_dim"]
        dim_k = params["dim_k"]

        self.linear_layers = []
        self.attention_layers = []
        for _ in range(h):
            W_Q = torch.nn.Linear(model_dim, dim_k, bias=False)
            W_K = torch.nn.Linear(model_dim, dim_k, bias=False)
            W_V = torch.nn.Linear(model_dim, dim_v, bias=False)
            self.linear_layers.append((W_Q, W_K, W_V))
            self.attention_layers.append(ScaledDotProductAttention(params))

        self.W_O = torch.nn.Linear(h * dim_v, model_dim, bias=False)


class Block(torch.nn.Module):
    def __init__(self, params):
        super(Block, self).__init__()

        self.multihead_attention = MultiHeadAttention(params)
        self.feed_forward = FeedForward(params)
        self.ln1 = torch.nn.LayerNorm(params["model_dim"])
        self.ln2 = torch.nn.LayerNorm(params["model_dim"])

    def forward(self, x):
        x_norm = self.ln1(x)
        x = x + self.multihead_attention(x_norm, x_norm, x_norm)
        x_norm = self.ln2(x)
        x = x + self.feed_forward(x_norm)
        return x


class LanguageModel(torch.nn.Module):
    def __init__(self, params):
        super(LanguageModel, self).__init__()

        self.params = params

        model_dim = params["model_dim"]
        self.embedding = torch.nn.Embedding(params["token_dim"], model_dim)
        self.pos_embedding = torch.nn.Embedding(params["context_length"], model_dim)
        self.blocks = torch.nn.Sequential(*[Block(params) for _ in range(params["transformer_blocks"])] + [torch.nn.LayerNorm(model_dim)])
        self.output = torch.nn.Linear(model_dim, params["token_dim"])
        self.register_buffer("position", torch.arange(params["context_length"]))


    def forward(self, x, targets=None):
        B,T = x.shape
        x = self.embedding(x) # batch, context_length, model_dim
        x = x + self.pos_embedding(self.position[:T])
        x = self.blocks(x)
        logits = self.output(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss

    
    def generate(self, x, n_tokens: int = 100):
        for _ in tqdm(range(n_tokens), unit="token"):
            # Truncate so we remain within context length
            truncated_x = x[:, -self.params["context_length"]:]
            logits, _ = self(truncated_x)
            # Softmax
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, token], dim=1)
        return x
