import pandas as pd
from transformers import AutoTokenizer
import torch

# Load the RWKV_TOKENIZER
class RWKV_TOKENIZER():
    # ... (the RWKV_TOKENIZER class definition you provided)
    pass

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
