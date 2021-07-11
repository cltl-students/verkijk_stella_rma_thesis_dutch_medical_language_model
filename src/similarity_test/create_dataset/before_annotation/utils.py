import pickle
from class_definitions import Annotation, BertContainer
from pathlib import Path
import torch
from datetime import datetime
import pandas as pd
import openpyxl
import numpy as np

def lightweightDataframe(data):
    """
    Creates dataframe with encodings and annotations per sentence
    (Encodings = last 4 layers of BERTje representation)
    Returns pandas dataframe
    
    :param data: pickle file
    """
    df_list = []   
    encodings = []
    for instance in data:
        #add encodings
        list_instance = []
        list_instance.append(instance.encoding)
        #make seperate list with encodings
        encodings.append(instance.encoding)
        
        #add lsit of labels per sentence
        list_labels = []
        annot = instance.annot
        for anno in annot:
            label = anno.label
            list_labels.append(label)
        list_instance.append(list_labels)
        df_list.append(list_instance)
        
    #create dataframe    
    df = pd.DataFrame(df_list, columns = ['encoding' , 'labels'])
    # create seperate columns for seperate labels
    df['disregard'] = 0
    df['background'] = 0
    df['target'] = 0
    df['viewthird'] = 0
    df['infothird'] = 0
    df['viewpatient'] = 0
    df['implicit'] = 0
    df['domain'] = 0
    df['delete'] = 0
    
    #Add 1 to certain columns
    df['disregard'][df['labels'].apply(lambda x: 'disregard\_file' in x)] = 1
    df['background'][df['labels'].apply(lambda x: 'type\_Background' in x)] = 1
    df['target'][df['labels'].apply(lambda x: 'target' in x)] = 1
    df['viewthird'][df['labels'].apply(lambda x: 'view\_Third party' in x)] = 1
    df['infothird'][df['labels'].apply(lambda x: 'info\_Third party' in x)] = 1
    df['viewpatient'][df['labels'].apply(lambda x: 'view\_Patient' in x)] = 1
    df['implicit'][df['labels'].apply(lambda x: 'type\_Implicit' in x)] = 1
    
    #Add domain labels to a seperate column
    df['domain'][df['labels'].apply(lambda x: '.D450: Lopen en zich verplaatsen' in x)] = '.D450: Lopen en zich verplaatsen'
    df['domain'][df['labels'].apply(lambda x: '.B455: Inspanningstolerantie' in x)] = '.B455: Inspanningstolerantie'
    df['domain'][df['labels'].apply(lambda x: '.D840-859: Beroep en werk' in x)] = '.D840-859: Beroep en werk'
    df['domain'][df['labels'].apply(lambda x: '.B152: Stemming' in x)] = '.B152: Stemming'

    #Add 1's to any row you want to delete
    df.loc[df['disregard'] == 1, 'delete'] = 1
    df.loc[df['background'] == 1, 'delete'] = 1
    df.loc[df['target'] == 1, 'delete'] = 1
    df.loc[df['viewthird'] == 1, 'delete'] = 1
    df.loc[df['infothird'] == 1, 'delete'] = 1
    df.loc[df['viewpatient'] == 1, 'delete'] = 1
    df.loc[df['domain'] == 0, 'domain'] = 'None'
    
    return(df, df_list)

def completeDataframe(data): 
    """
    Creates dataframe with key, annoator, sentence id, sentence in natural language, sentence representation in BERTje encoding, and annotations
    Returns pandas dataframe
    
    :param data: pickle file
    """
    df_list = []    
    for instance in data:
        list_instance = []
        list_instance.append(instance.key)
        list_instance.append(instance.annotator)
        list_instance.append(instance.sen_id)
        list_instance.append(instance.sen)
        list_instance.append(instance.encoding)
        #add labels
        list_labels = []
        annot = instance.annot
        for anno in annot:
            label = anno.label
            list_labels.append(label)
        list_instance.append(list_labels)
        df_list.append(list_instance)
    df = pd.DataFrame(df_list, columns = ['key', 'annotator', 'sen_id', 'sen', 'encoding', 'labels'])
      
    df['disregard'] = 0
    df['background'] = 0
    df['target'] = 0
    df['viewthird'] = 0
    df['infothird'] = 0
    df['viewpatient'] = 0
    df['implicit'] = 0
    df['domain'] = 0
    df['delete'] = 0

    df['disregard'][df['labels'].apply(lambda x: 'disregard\_file' in x)] = 1
    df['background'][df['labels'].apply(lambda x: 'type\_Background' in x)] = 1
    df['target'][df['labels'].apply(lambda x: 'target' in x)] = 1
    df['viewthird'][df['labels'].apply(lambda x: 'view\_Third party' in x)] = 1
    df['infothird'][df['labels'].apply(lambda x: 'info\_Third party' in x)] = 1
    df['viewpatient'][df['labels'].apply(lambda x: 'view\_Patient' in x)] = 1
    df['implicit'][df['labels'].apply(lambda x: 'type\_Implicit' in x)] = 1
    df['domain'][df['labels'].apply(lambda x: '.D450: Lopen en zich verplaatsen' in x)] = '.D450: Lopen en zich verplaatsen'
    df['domain'][df['labels'].apply(lambda x: '.B455: Inspanningstolerantie' in x)] = '.B455: Inspanningstolerantie'
    df['domain'][df['labels'].apply(lambda x: '.D840-859: Beroep en werk' in x)] = '.D840-859: Beroep en werk'
    df['domain'][df['labels'].apply(lambda x: '.B152: Stemming' in x)] = '.B152: Stemming'
    
    df.loc[df['disregard'] == 1, 'delete'] = 1
    df.loc[df['background'] == 1, 'delete'] = 1
    df.loc[df['target'] == 1, 'delete'] = 1
    df.loc[df['viewthird'] == 1, 'delete'] = 1
    df.loc[df['infothird'] == 1, 'delete'] = 1
    df.loc[df['viewpatient'] == 1, 'delete'] = 1
    df.loc[df['domain'] == 0, 'domain'] = 'None'
    
    return(df, df_list)

def filterDataframe(df):
    """
    Takes out the rows that need to be deleted from df: background, target, view_patient, view_thirdparty, info_thirdparty
    Returns filtered df and list of indexes of rows that were deleted
    
    :param df: pandas dataframe
    """
    rows_to_delete = []
    for index, item in enumerate(df['delete']):
        if item == 1:
            rows_to_delete.append(index)
    df = df.drop(rows_to_delete)
    return(rows_to_delete, df)

        

