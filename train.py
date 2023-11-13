from datasets import load_dataset
import torch

from tokenizer import build_dictionary, tokenize, untokenize
from transformer import MultiHeadAttention

dataset = load_dataset("tiny_shakespeare")

train_text = dataset["train"]["text"][0]


dictionary = build_dictionary(train_text)
training_tokens = tokenize(train_text, dictionary).type(torch.float)

token_dim = len(dictionary)
embedding_dim = 512

token_embedder = torch.nn.Linear(token_dim, embedding_dim)
attention = MultiHeadAttention()

training_embeddings = token_embedder(training_tokens)
print(attention(training_embeddings, training_embeddings, training_embeddings))
