'''
Author:        Dipayan
Last updated:  2025-06-10
Licence:       MIT (see LICENCE file)
Description:  This script loads the ProtT5 model and tokenizer, processes protein sequences from a CSV file, computes their embeddings, and saves the embeddings to a .pth file.
'''

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Using device: {}".format(device))

transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
T5model = T5EncoderModel.from_pretrained(transformer_link)
T5model.full() if device=='cpu' else T5model.half() # only cast to full-precision if no GPU is available
T5model = T5model.to(device)
T5model = T5model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

print("Model and tokenizer loaded successfully!")

import re

def embedding_dict_values(sequences):
    sequence_examples = sequences
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    embeddings_dict = {}
    for idx, sequence in enumerate(sequence_examples):
        ids = tokenizer.encode_plus(sequence, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = T5model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = embedding.last_hidden_state.squeeze(0) 
        mask = attention_mask.squeeze(0).bool()  
        filtered_embeddings = token_embeddings[mask]
        embeddings_dict[sequences[idx]] = filtered_embeddings.cpu()
        print(f"Processed sequence {idx + 1}/{len(sequence_examples)}")
        filtered_embeddings = None
    return embeddings_dict

## Add path to the Processsed CSV files##
PN = pd.read_csv(  r"../example.csv")

mat_dict = embedding_dict_values(sequences=PN["col1"].values.tolist()  + PN["col2"].values.tolist())

### Save the embeddings dictionary to a file with .pth extension ###
save_path = r"../example.pth"
torch.save(mat_dict, save_path)

#########################################################################################



































