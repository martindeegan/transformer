import torch
import numpy as np

def build_dictionary(input_text):
    dictionary = sorted(list(set(input_text)))
    stoi = {c: i for i, c in enumerate(dictionary)}
    itos = {i: c for i, c in enumerate(dictionary)}
    return stoi, itos


def tokenize(input_text: str, dictionary) -> torch.Tensor:
    stoi, _ = dictionary
    return torch.tensor([stoi[c] for c in input_text], dtype=torch.long)


def detokenize(input_arr: torch.Tensor, dictionary) -> str:
    _, itos = dictionary
    return "".join([itos[i.item()] for i in input_arr.squeeze(0)])