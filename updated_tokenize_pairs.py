import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np

# Load the RWKV_TOKENIZER
class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[max(i,1)], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()


from transformers import AutoTokenizer

neox = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

rwkv_tokenizer = RWKV_TOKENIZER("1B5.pth")  # Replace with the actual path to your RWKV model

# Load the merged translations CSV file
merged_df = pd.read_csv('all_translation_pairs.csv')

# Create an empty DataFrame to store tokenized translations
tokenized_df = pd.DataFrame()

# Loop through columns and tokenize each translation pair
for column in merged_df.columns:
    # Tokenize each translation using RWKV_TOKENIZER
    rwkv_tokens = [rwkv_tokenizer.encode(text) for text in merged_df[column]]

    # Convert RWKV tokens to strings
    rwkv_tokens_as_strings = [
        ' '.join([rwkv_tokenizer.idx2token[token].decode('utf-8') for token in tokens])
        for tokens in rwkv_tokens
    ]

    # Join the tokenized translations into a single string
    tokenized_df[column] = rwkv_tokens_as_strings

# Load a Transformers model (e.g., gpt-neox-20b)
neox = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Tokenize the tokenized_df using the Transformers tokenizer
for column in tokenized_df.columns:
    tokenized_df[column] = tokenized_df[column].apply(neox.tokenize)

# Save the tokenized DataFrame to a new CSV file
tokenized_df.to_csv('tokenized_translations.csv', index=False)

# Optionally, you can convert the tokenized data to PyTorch tensors or use it as needed.

# Converting the tokenized DataFrame to a NumPy array
tokenized_array = tokenized_df.to_numpy()

# Saving the tokenized data as a NumPy file
np.save('tokenized_data.npy', tokenized_array)
