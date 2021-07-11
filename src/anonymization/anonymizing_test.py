"""
@Author StellaVerkijk
This script gathers predictions of the From Scratch Language Model for a data set of sentences where names are masked.
Filepaths are hardcoded since the data used could not be released to the public because of privacy issues
"""

import transformers
from transformers import pipeline, RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModel
from collections import Counter
import pandas as pd
import pickle

print("Loading model...")
tokenizer = RobertaTokenizer.from_pretrained("../../processing/from_scratch_final_model_new_vocab")
model = RobertaForMaskedLM.from_pretrained("../../processing/from_scratch_final_model_new_vocab")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

print("Getting sentences...")
df = pd.read_csv("anon_specific_testset_eval.csv", delimiter = ';')
list_sens = df['sentences'].tolist()

list_of_ds = []
for sen in list_sens:
    d = dict()
    d['sen'] = sen
    list_of_ds.append(d)
    
print("Making predictions...")
i = 0
for d in list_of_ds:
    i+=1
    pre_dicts = fill_mask(d['sen'], top_k=20)
    tokens = []
    for pred in pre_dicts:
        tokens.append(pred['token_str'])
    d['predictions'] = tokens
        

print("Adding all predictions together...")
all_predictions = []
for d in list_of_ds:
    for item in d['predictions']:
        all_predictions.append(item)
        
print(len(all_predictions))
        
print("Counting predictions...")
from collections import Counter
counts = Counter(all_predictions)

df = pd.DataFrame(list(counts.items()),columns = ['Prediction','times predicted'])
df = df.sort_values(by='times predicted', ascending=False)

df.to_csv("predictions_from_scratch_unseen_data.csv", sep = ';', index = None)
