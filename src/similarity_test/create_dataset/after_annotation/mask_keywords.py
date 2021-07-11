"""
Author @ Stella Verkijk
mask keywords from similarity testset

run from commandline with three arguments
- path to similarity test dataset 
- path and filename to where you want the data set with masked keywords for a roberta-based model stored 
- path and filename to where you want the data set with masked keywords for a bert-based model stored 
example run:
python mask_keywords.py '../../data/complete_simtest.csv' '../../data/complete_simtest_no_keywords_roberta.csv' '../../data/complete_simtest_no_keywords_bert.csv'
"""

import pandas as pd
import sys

keywords_lopen = [" FAC0 ", " FAC1 ", " FAC2 ", " FAC3 ", " FAC4 ", " FAC5 ", " FAC 0 ", " FAC 1 ", " FAC 2 ", " FAC 3 ", " FAC 4 ", " FAC 5 ", " Transfer ", " transfer ", "mobiliteit", "Mobiliteit", " tillift ", " Tillift ", " rolstoel ", " Rolstoel ", " stoel ", " Stoel ", " bed ", " Bed ", " stapjes ", " Stapjes ", " stap ", " Stap ", " stappen ", " Stappen "]

keywords_stemming = ['emotioneel', 'Emotioneel', 'droevig', 'Droevig', 'verdrietig', 'Verdrietig', 'huilt', 'Huilt', 'huilen', 'Huilen','blij', 'Blij', 'tevreden', 'Tevreden', 'rustig', 'Rustig', 'onrustig', 'Onrustig', 'apatisch', 'Apatisch', 'verward', 'Verward', " modulerend affect ", " affect vlak ", " somber ", " niet blij ", " vrolijk "]

keywords_beroep = ['kantoor', 'Kantoor', 'bouw', 'Bouw', 'niet naar school', 'Niet naar school', ' les ', ' Les ']

keywords_inspanning = [' saturatie ', ' Saturatie ', ' saturatiedip ', ' Saturatiedip ', 'conditie', 'Conditie', 'snel vermoeid', 'Snel vermoeid', 'vermoeid', 'Vermoeid', 'uitgeput', 'Uitgeput', 'snel moe', 'Snel moe', ' saturatie dip ', ' sport ', ' Sport ']

keywords = keywords_lopen + keywords_stemming + keywords_beroep + keywords_inspanning
print(keywords)


def gather_sentences(path_to_dataset):
    
    simtest = pd.read_csv(path_to_dataset, sep = '\t', encoding = 'utf-8')

    sents1 = simtest['Sen1'].tolist()
    sents2 = simtest['Sen2'].tolist()
    sents3 = simtest['Sen3'].tolist()
    annotations = simtest['Annotation'].tolist()
    ids = simtest['IDs'].tolist()
    
    return(sents1, sents2, sents3, annotations, ids)


def mask_keywords(sents1, sents2, sents3, annotations, ids, modeltype): 
    
    if modeltype == 'roberta':
        mask = ' <mask> '
    if modeltype == 'bert':
        mask = ' [MASK] '
        
        
    sents1_nk = []
    for sentence in sents1:
        for word in keywords:
            if word in sentence:
                sentence = sentence.replace(word, mask)
        sents1_nk.append(sentence)
                
    print(len(sents1_nk))
    
    sents2_nk = []
    for sentence in sents2:
        for word in keywords:
            if word in sentence:
                sentence = sentence.replace(word, mask)
        sents2_nk.append(sentence)

            
    sents3_nk = []
    for sentence in sents3:
        for word in keywords:
            if word in sentence:
                sentence = sentence.replace(word, mask)
        sents3_nk.append(sentence)

            
    simtest_nk = pd.DataFrame()
    simtest_nk['Sen1'] = sents1_nk
    simtest_nk['Sen2'] = sents2_nk    
    simtest_nk['Sen3'] = sents3_nk
    simtest_nk['Annotation'] = annotations
    simtest_nk['IDs'] = ids
    
    return(simtest_nk)


def main(path_to_dataset, outfile1, outfile2):

    sents1, sents2, sents3, annotations, ids = gather_sentences(path_to_dataset)
    simtest_nk_roberta = mask_keywords(sents1, sents2, sents3, annotations, ids, 'roberta')
    simtest_nk_roberta.to_csv(outfile1, sep =';', index = False, encoding = 'utf-8')
    simtest_nk_bert = mask_keywords(sents1, sents2, sents3, annotations, ids, 'bert')
    simtest_nk_bert.to_csv(outfile2, sep =';', index = False, encoding = 'utf-8')
    
path_to_dataset = sys.argv[1]
outfile1 = sys.argv[2]
outfile2 = sys.argv[3]
main(path_to_dataset, outfile1, outfile2)
