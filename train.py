import os
import logging
import pickle
import argparse

from datasets import load_dataset
import torch

from tokenizer import build_dictionary, tokenize, detokenize
from transformer import LanguageModel, default_params

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

if not os.path.exists("data/train.pt") or not os.path.exists("data/val.pt"):
    logging.info("Tokenized dataset not found. Generating...")

    dataset = load_dataset("tiny_shakespeare")
    train_text = dataset["train"]["text"][0]
    val_text = dataset["validation"]["text"][0]

    dictionary = build_dictionary(train_text + val_text)
    train = tokenize(train_text, dictionary)
    val = tokenize(val_text, dictionary)

    # Save the tokenized dicts
    os.makedirs("data", exist_ok=True)
    torch.save(train, "data/train.pt")
    torch.save(val, "data/val.pt")
    with open("data/dictionary.pkl", "wb") as f:
        pickle.dump(dictionary, f)
else:
    logging.info("Loading pre-tokenized dataset...")
    train = torch.load("data/train.pt")
    val = torch.load("data/val.pt")
    dictionary = pickle.load(open("data/dictionary.pkl", "rb"))


def get_batch(split: torch.Tensor, batch_size: int = 4, context_length: int = 8):
    ix = torch.randint(len(split) - context_length, (batch_size,))
    x = torch.stack([split[i : i + context_length] for i in ix])
    y = torch.stack([split[i + 1 : i + context_length + 1] for i in ix])
    return x, y


params = default_params()
params["context_length"] = 256
params["token_dim"] = len(dictionary[0])


try:
    logging.info("Loading model...")
    lm = LanguageModel(params)
    lm.load_state_dict(torch.load("data/model.pt"))
except:
    logging.info("Loading model failed. Creating new model...")
    lm = LanguageModel(params)
    torch.save(lm.state_dict(), "data/model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lm.to(device)


if not args.train:
    lm.eval()
    print("-----------------------------------------------")
    print("Running inference")
    print("-----------------------------------------------")
    x, _ = get_batch(val, batch_size=1, context_length=params["context_length"])
    x = x.to(device)
    output = lm.generate(x[:1], n_tokens=5000)
    print(detokenize(output.cpu(), dictionary))
else:

    lm = lm.train()

    batch_size = 64
    max_iters = 10000
    eval_interval = 100
    optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-4)
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        # if iter % eval_interval == 0 or iter == max_iters - 1:
        #     losses = estimate_loss()
        #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train, batch_size, context_length=params["context_length"])
        xb = xb.to(device)
        yb = yb.to(device)

        # evaluate the loss
        logits, loss = lm(xb, yb)
        print(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(lm.state_dict(), "data/model.pt")


