import torch
import numpy as np

def build_dictionary(input_text):
    dictionary = set(input_text)
    print("Dictionary:\n==================================")
    print(repr("".join(sorted(dictionary))))
    print("==================================")
    dictionary = sorted([ord(c) for c in dictionary])
    return dictionary


def build_lookup(dictionary):
    # Find characters
    max_token = max(dictionary)

    # Build lookup table
    lookup = -torch.ones(max_token + 1, dtype=torch.int64)
    for i, token in enumerate(dictionary):
        lookup[token] = i

    return lookup


# Tokenize text
def tokenize(input_text, dictionary):
    token_dim = len(dictionary)
    lookup = build_lookup(dictionary)
    input_arr = np.array(list(input_text))
    text_as_int = input_arr.view(np.int32)

    # Lookup each value
    text_lookup = lookup[text_as_int]
    return torch.nn.functional.one_hot(text_lookup, token_dim)


def untokenize(one_hot_encoding, dictionary):
    labels = torch.argmax(one_hot_encoding, axis=1)

    lookup = build_lookup(dictionary)
    inv_lookup = -torch.ones_like(lookup)
    for i, x in enumerate(lookup):
        if x >= 0:
            inv_lookup[x] = i

    output = inv_lookup[labels]
    output = np.array(output, dtype=np.int32)
    return "".join(output.view('U1'))
