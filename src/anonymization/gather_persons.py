"""
@Author StellaVerkijk
This script creates a dataset of 8000 sentences that can be used for an anonymization test using the fill-mask task of transformer language models.
For the anonymization test of the From Scratch Language model, a subset of the data set created with this script was used.
This script contains one function that loads a text file, splits on sentences, select sentences that contain the token 'PERSON' and that are between 30 and 120 characters long, replaces the PERSON tokens in these sentences with the <mask> token and writes a sample of 8000 of these sentences to a csv file.
"""

import glob
from pathlib import Path
import pandas as pd
from random import sample

def gather_sentences(path_to_textfile, outdir):
    """
    Loads a text file, splits on sentences, select sentences that contain the token 'PERSON' and that are between 30 and 120 characters long, replaces the PERSON tokens in these sentences with the <mask> token and writes a sample of 8000 of these sentences to a csv file.
    
    :param path_to_textfile: str
    :param outdir: str
    """
    
    #select sentences with PERSON that are longer than 30 characters, replace PERSON with <mask> and gather the sentences in a list
    sentences = []
    with open (path_to_textfile, 'r', encoding = 'utf-8') as infile:
        for line in infile.readlines():
            sens = line.split('. ')
            for sen in sens:
                if 'PERSON' in sen:
                    if len(sen) > 30:
                        sentences.append(sen.replace('PERSON', '<mask>').strip('\n')) 

    #remove sentences from the list that contain more than 120 characters and add a full stop to each sentence
    filtered_sentences = []
    for sentence in sentences:
        if len(sentence) >= 30 and len(sentence) <= 120:
            filtered_sentences.append(sentence + '.')
            
    #remove sentences that contain more than one <mask> token
    for sentence in filtered_sentences:
        tokens = sentence.split()
        i = 0
        for token in tokens:
            if '<mask>' in token:
                i+=1
        if i > 1: 
            filtered_sentences.remove(sentence)
        i = 0
        
    #take a random sample of 8000 sentences of the current selection
    random = sample(filtered_sentences, 8000)
    
    #write to a dataframe
    df_small = pd.DataFrame()
    df_small['sentences'] = random
    df_small['guessed_tokens'] = 0

    
    df_small.to_csv(outdir, sep = ';', index = False)
                        
                     
#gather_sentences("../gather_traindata/data/anonymised/validation/eval.txt", "anon_testset_eval.csv")
