"""
@Author StellaVerkijk
this script supports the create_triples.py script
gathers all sentences annotated with an ICF domain from the a-proof data
"""

import pickle
import pandas as pd
from utils import completeDataframe, filterDataframe

def createDataframe(filepath):
    """
    Creates dataframe of all sentences annotated with an ICF domain
    """
    domains = ['.D450: Lopen en zich verplaatsen', '.B152: Stemming', '.D840-859: Beroep en werk', '.B455: Inspanningstolerantie']
    print("Reading pickle files...")
    #read pickle files
    with open(filepath, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    df, df_info = completeDataframe(data)
    
    print("Selecting annotated sentences...")
    df_selection = df.loc[df['domain'].isin(domains)]
    
    return(df_selection)


def sen_per_domain(df, domain):
    """
    Creates lists of sentences annotated with a specific domain per list
    """
    domains = ['.D450: Lopen en zich verplaatsen', '.B152: Stemming', '.D840-859: Beroep en werk', '.B455: Inspanningstolerantie']
    
    if domain == 'lopen':
        d = '.D450: Lopen en zich verplaatsen'
    elif domain == 'stemming':
        d = '.B152: Stemming'
    elif domain == 'beroep':
        d = '.D840-859: Beroep en werk'
    elif domain == 'inspanningstolerantie':
        d = '.B455: Inspanningstolerantie'
        
    sentences = []
    for index, item in enumerate(df['domain']):
        if item == d:
            sentences.append(df.iloc[index]['sen'])

    return(sentences)

def all_sentences(df):
    """
    Returns a list of sentences annotated with any domain
    """
    sentences = df['sen'].tolist()
    return(sentences)


def createDataframeLevel(filepath_train, domain):
    """
    Creates dataframe with encodings and annotations per sentence
    (Encodings = last 4 layers of BERTje representation)
    Returns pandas dataframe

    :param domain: str: lopen, stemming, beroep, inspanningstolerantie
    :param type_dataset: str: covid, noncovid
    """
    
    print("Reading pickle files...")
    #read pickle files
    with open(filepath_train, "rb") as pkl_file:
        traindata = pickle.load(pkl_file)
    
    if domain == 'lopen':
        d = '.D450: Lopen en zich verplaatsen'
        l = 'FAC '
    elif domain == 'stemming':
        d = '.B152: Stemming'
        l = 'STM '
    elif domain == 'beroep':
        d = '.D840-859: Beroep en werk'
        l = 'BER '
    elif domain == 'inspanningstolerantie':
        d = '.B455: Inspanningstolerantie'
        l = 'INS '
    
    df_train, df_list_tr = completeDataframe(traindata)
    
    
    df_train[domain] = 0
    df_train['level'] = 'None'

    
    #Add domain labels to a seperate column
    df_train['domain'][df_train['labels'].apply(lambda x: d in x)] = d
    df_train['level'][df_train['labels'].apply(lambda x: l+'0'in x)] = 0
    df_train['level'][df_train['labels'].apply(lambda x: l+'1'in x)] = 1
    df_train['level'][df_train['labels'].apply(lambda x: l+'2'in x)] = 2
    df_train['level'][df_train['labels'].apply(lambda x: l+'3'in x)] = 3
    df_train['level'][df_train['labels'].apply(lambda x: l+'4'in x)] = 4
    df_train['level'][df_train['labels'].apply(lambda x: l+'5'in x)] = 5
    
    df_train.loc[df_train['domain'] == d, domain] = 1
    
        
    print("Filtering dataframes...")
    #filter dataframe
    del_rows_tr, filtered_df_train = filterDataframe(df_train)
    #only select instances where there is an entry for level
    df_selection_train = filtered_df_train[(filtered_df_train[domain] == 1) & (filtered_df_train['level'] != 'None')]

    return(df_selection_train)


