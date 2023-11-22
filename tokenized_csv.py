import pandas as pd

import src.tokenizer as tk
import tqdm
import numpy as np
from translate.storage.tmx import tmxfile

# sample of the tmx file
# <?xml version="1.0" encoding="UTF-8" ?>
# <tmx version="1.4">
# <header creationdate="Thu Oct 12 19:01:10 2017"
#           srclang="en"
#           adminlang="en"
#           o-tmf="unknown"
#           segtype="sentence"
#           creationtool="Uplug"
#           creationtoolversion="unknown"
#           datatype="PlainText" />
#   <body>
#     <tu>
#       <tuv xml:lang="en"><seg>I never dreamed before</seg></tuv>
#       <tuv xml:lang="fr"><seg>I've never dreamed before I'm gonna knock the door</seg></tuv>
#     </tu>
#     <tu>
#       <tuv xml:lang="en"><seg>I'm gonna knock the door into the world of perfect free (You ain't no lonely!</seg></tuv>
#       <tuv xml:lang="fr"><seg>Into the world of perfect free You ain't no lonely !</seg></tuv>
#     </tu>
#     <tu>
#       <tuv xml:lang="en"><seg>You're gonna say I'm lying I'm gonna get the chance</seg></tuv>
#       <tuv xml:lang="fr"><seg>You're gonna say I'm lying I'm gonna get the chance</seg></tuv>
#     </tu>

# def load_tmx_file(file_path):
#     current_batch = []
    
#     # Open the file for saving the tokenized pairs
#     with open(file_path + ".py", 'ab') as f:
#         # Load the TMX file
#         tmx = tmxfile(file_path)
        
#         # Iterate over the translation units in the TMX file
#         for tu in tmx.unit_iter():
#             # Get the source and target texts
#             source_text = tu.source.encode('utf-8')
#             target_text = tu.target.encode('utf-8')
#             print(source_text, target_text)
#             # Tokenize the pair and add to the current batch
#             tokenized_pair = tk.world.encode("\nSource: " + source_text + "\nTarget: " + target_text) + [0]
#             current_batch.append(tokenized_pair)
            
#             # Save the batch if it reaches a certain size
#             if len(current_batch) >= 10000:
#                 # Flatten the list of tokenized pairs
#                 flat_list = [item for sublist in current_batch for item in sublist]
#                 # Append to the file
#                 np.save(f, flat_list)
#                 # Clear the current batch
#                 current_batch = []
        
#         # Save any remaining pairs in the last batch
#         if len(current_batch) > 0:
#             flat_list = [item for sublist in current_batch for item in sublist]
#             np.save(f, flat_list)

# load_tmx_file("en-fr.tmx")


def load_tmx_file(file_path):
    current_batch = []
    
    # Open the file for saving the tokenized pairs
    with open(file_path + ".py", 'ab') as f:
        # Load the TMX file

        with open(file_path, 'rb') as fin:
            # Load the TMX file
            tmx = tmxfile(fin, 'en', 'fr')
        
        # Iterate over the translation units in the TMX file
        for tu in tmx.unit_iter():
            print("tu is loaded")
            # Get the source and target texts
            source_text = tu.source
            target_text = tu.target
            print(source_text, target_text)
            # Tokenize the pair and add to the current batch
            tokenized_pair = tk.world.encode("\nSource: " + source_text + "\nTarget: " + target_text) + [0]
            current_batch.append(tokenized_pair)
            
            # Save the batch if it reaches a certain size
            if len(current_batch) >= 10000:
                # Flatten the list of tokenized pairs
                flat_list = [item for sublist in current_batch for item in sublist]
                # Append to the file
                np.save(f, flat_list)
                # Clear the current batch
                current_batch = []
        
        # Save any remaining pairs in the last batch
        if len(current_batch) > 0:
            flat_list = [item for sublist in current_batch for item in sublist]
            np.save(f, flat_list)

load_tmx_file("en-fr.tmx")