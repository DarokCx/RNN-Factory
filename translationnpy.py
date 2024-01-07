
import pandas as pd
import numpy as np
from itertools import permutations
import src.tokenizer as tk

def load_csv_files(file_paths):
    combined_data = pd.DataFrame(columns=['Source Language', 'Target Language', 'Source Text', 'Target Text'])
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        source_lang = 'en'
        target_lang = file_path.split('/')[-1].split('-')[1].split('.')[0]
        combined_data = pd.concat([combined_data, df[['Input', 'Response']].rename(columns={'Input': 'Source Text', 'Response': 'Target Text'})], ignore_index=True)
        combined_data.loc[combined_data['Target Text'].notnull(), 'Source Language'] = source_lang
        combined_data.loc[combined_data['Target Text'].notnull(), 'Target Language'] = target_lang
    return combined_data

def create_language_permutations(combined_data):
    swapped_data = combined_data.copy()
    swapped_data.columns = ['Target Language', 'Source Language', 'Target Text', 'Source Text']
    permutations_data = pd.concat([combined_data, swapped_data], ignore_index=True)
    permutations_data = permutations_data[permutations_data['Source Language'] != permutations_data['Target Language']]
    return permutations_data

def tokenize_and_save(permutations_data, output_file):
    tokenized_batches = []
    for index, row in permutations_data.iterrows():
        source_text = row['Source Text']
        target_text = row['Target Text']
        tokenized_pair = tk.world.encode("\nSource: " + source_text + "\nTarget: " + target_text) + [0]
        tokenized_batches.append(tokenized_pair)

        # Assuming a batch size of 10000 for saving
        if len(tokenized_batches) >= 10000:
            flat_list = [item for sublist in tokenized_batches for item in sublist]
            np.save(output_file, flat_list, allow_pickle=True)
            tokenized_batches = []

    # Saving any remaining pairs in the last batch
    if len(tokenized_batches) > 0:
        flat_list = [item for sublist in tokenized_batches for item in sublist]
        np.save(output_file, flat_list, allow_pickle=True)

file_paths = [
    'en-fr.csv',
    'en-it.csv',
    'en-ch.csv',
    'en-ger.csv'
]

combined_data = load_csv_files(file_paths)
permutations_data = create_language_permutations(combined_data)
tokenize_and_save(permutations_data, 'output_tokenized_data.npy')
