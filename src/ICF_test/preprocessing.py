"""
@author StellaVerkijk
"""

import pickle
from class_definitions import Annotation, BertContainer
from pathlib import Path
import torch
from datetime import datetime
import pandas as pd
import openpyxl
import numpy as np
from utils import lightweightDataframe, completeDataframe, filterDataframe

def prepareDataC(function, splitting = False, percentage = 100):   
    """
    Reads covid train and test data
    Prepares dataframes
    Retrieves note id's
    Outputs train data and labels and test data and labels and note id's
    
    :param function: variable: completeDtaframe or lightweightDataframe
    :param splitting: bool: when taking only a percentage of the covid data, choose True
    :param percentage: int: if splitting = True, indicate percentage of covid data you want: 25, 50 or 75
    """
    print(datetime.now())
    #define paths to train and test pickles    
    input_train_covid = '../../Covid_data_11nov/a-proof-exp/traindata_covidbatch.pkl'
    input_test_covid =  '../../Covid_data_11nov/a-proof-exp/testdata_covidbatch.pkl' 
    
    print("Reading pickle files...")
    #read pickle files
    with open(input_train_covid, "rb") as pkl_file:
        traindata_c = pickle.load(pkl_file)
        
    with open(input_test_covid, "rb") as pkl_file:
        testdata_c = pickle.load(pkl_file)
        
        
    print("Creating and filtering dataframes...")
    #prepare training dataframes
    df_tr_c = function(traindata_c)[0]
    #take out sentences with labels that we should ignore (background, target, view_patient, view_thirdparty, info_thirdparty)
    rows_to_delete_tr_c, filtered_df_tr_c = filterDataframe(df_tr_c)

    #prepare test dataframes
    df_te_c = function(testdata_c)[0]
    rows_to_delete_te_c, filtered_df_te_c = filterDataframe(df_te_c)
    #extract test labels
    filtered_labels_te_c = filtered_df_te_c['domain'].to_list()
    filtered_levels_te_c = filtered_df_te_c['level'].to_list()
    filtered_sentences_te_c = filtered_df_te_c['sen'].tolist()
    filtered_sen_ids_te_c = filtered_df_te_c['sen_id'].tolist()

    print("Retrieve note id's...")
    #get note id's for aggregation
    try:    
        ids_c = []
        list_keys_c = filtered_df_te_c['key'].tolist()
        for key in list_keys_c:
            y = key.split('--')[3]
            ids_c.append(y)
    except KeyError:
        ids_c = []
        
    
    with open("down_indices_covid2.pkl", "rb") as f:
        indices = pickle.load(f)
    down_df_tr_c = filtered_df_tr_c.drop(indices)
    
    
    if splitting == False:
        downsampled_filtered_labels_tr_c = down_df_tr_c['domain'].to_list()
        downsampled_filtered_levels_tr_c = down_df_tr_c['level'].to_list()
        downsampled_filtered_sentences_tr_c = down_df_tr_c['sen'].tolist()
        downsampled_filtered_sen_ids_tr_c = down_df_tr_c['sen_id'].tolist()
        
    if splitting == True:
        #splitting final dataframw
        shuffled = down_df_tr_c.sample(frac = 1)
        parts = np.array_split(shuffled, 4)
                               
        df_25 = parts[0]
        df_50 = df_25.append(parts[1])
        df_75 = df_50.append(parts[2])
        
        if percentage == 25:
            #extract training labels
            downsampled_filtered_labels_tr_c = df_25['domain'].to_list()
            downsampled_filtered_encodings_tr_c = df_25['encoding'].tolist()
        if percentage == 50:
            downsampled_filtered_labels_tr_c = df_50['domain'].to_list()
            downsampled_filtered_encodings_tr_c = df_50['encoding'].tolist()            
        if percentage == 75:
            downsampled_filtered_labels_tr_c = df_75['domain'].to_list()
            downsampled_filtered_encodings_tr_c = df_75['encoding'].tolist() 
        
    df_scriptie_tr_c = pd.DataFrame()
    df_scriptie_tr_c['sentence'] = downsampled_filtered_sentences_tr_c
    df_scriptie_tr_c['label'] = downsampled_filtered_labels_tr_c
    df_scriptie_tr_c['level'] = downsampled_filtered_levels_tr_c
    df_scriptie_tr_c['sen_id'] = downsampled_filtered_sen_ids_tr_c
    
    df_scriptie_te_c = pd.DataFrame()
    df_scriptie_te_c['sentence'] = filtered_sentences_te_c
    df_scriptie_te_c['label'] = filtered_labels_te_c
    df_scriptie_te_c['level'] = filtered_levels_te_c
    df_scriptie_te_c['sen_id'] = filtered_sen_ids_te_c 

    return(df_scriptie_tr_c, df_scriptie_te_c)

def prepareDataNC(function):
    """
    Reads non covid train and test data
    Prepares dataframes
    Retrieves note id's
    Outputs train data and labels and test data and labels and note id's
    
    :param function: variable: completeDtaframe or lightweightDataframe
    """
    print(datetime.now())
    #define paths to train and test pickles    
    input_train_noncovid = '../../Non_covid_data_15oct/a-proof-exp/train_data_batch1_disregard_removed.pkl'
    input_test_noncovid = '../../Non_covid_data_15oct/a-proof-exp/test_data_batch1_disregard_removed.pkl'
    
    print("Reading pickle files...")
    #read pickle files        
    with open(input_train_noncovid, "rb") as pkl_file:
        traindata_nc = pickle.load(pkl_file)
        
    with open(input_test_noncovid, "rb") as pkl_file:
        testdata_nc = pickle.load(pkl_file)
        
    print("Creating and filtering dataframes...")
    #prepare training dataframes
    df_tr_nc = function(traindata_nc)[0]
    #take out sentences with labels that we should ignore (background, target, view_patient, view_thirdparty, info_thirdparty)
    rows_to_delete_tr_nc, filtered_df_tr_nc = filterDataframe(df_tr_nc)

    #prepare test dataframes
    df_te_nc = function(testdata_nc)[0]
    rows_to_delete_te_nc, filtered_df_te_nc = filterDataframe(df_te_nc)
    #extract test labels
    filtered_labels_te_nc = filtered_df_te_nc['domain'].to_list()
    filtered_levels_te_nc = filtered_df_te_nc['level'].to_list()
    filtered_sentences_te_nc = filtered_df_te_nc['sen'].tolist()
    filtered_sen_ids_te_nc = filtered_df_te_nc['sen_id'].tolist()
    
    print("Retrieve note id's...")
    #get keys for aggregation
    try: 
        ids_nc = []
        df_list = function(testdata_nc)[1]
        df_list_new =  [i for j, i in enumerate(df_list) if j not in set(rows_to_delete_te_nc)]
        for instance in df_list_new:
            l = len(str(instance[2]))
            if instance[3] == "['']":
                ids_nc.append(instance[0].split('---')[1])
            else:
                ids_nc.append(instance[0].split('---')[1][:-l])
    except IndexError:
        ids_nc = []
                    
    
    with open("down_indices3.pkl", "rb") as f:
        indices = pickle.load(f)
    down_df_tr_nc = filtered_df_tr_nc.drop(indices)
    downsampled_filtered_labels_tr_nc = down_df_tr_nc['domain'].to_list()
    downsampled_filtered_levels_tr_nc = down_df_tr_nc['level'].to_list()
    downsampled_filtered_sentences_tr_nc = down_df_tr_nc['sen'].tolist()
    downsampled_filtered_sen_ids_tr_nc = down_df_tr_nc['sen_id'].tolist()

    df_scriptie_tr_nc = pd.DataFrame()
    df_scriptie_tr_nc['sentence'] = downsampled_filtered_sentences_tr_nc
    df_scriptie_tr_nc['label'] = downsampled_filtered_labels_tr_nc
    df_scriptie_tr_nc['level'] = downsampled_filtered_levels_tr_nc
    df_scriptie_tr_nc['sen_id'] = downsampled_filtered_sen_ids_tr_nc
    
    df_scriptie_te_nc = pd.DataFrame()
    df_scriptie_te_nc['sentence'] = filtered_sentences_te_nc
    df_scriptie_te_nc['label'] = filtered_labels_te_nc
    df_scriptie_te_nc['level'] = filtered_levels_te_nc
    df_scriptie_te_nc['sen_id'] = filtered_sen_ids_te_nc 

    return(df_scriptie_tr_nc, df_scriptie_te_nc, ids_nc)


df_tr_nc, df_te_nc, ids_nc  = prepareDataNC(completeDataframe)
df_tr_c, df_te_c = prepareDataC(completeDataframe)




