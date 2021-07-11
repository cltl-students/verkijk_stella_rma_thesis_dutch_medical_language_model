"""
@author StellaVerkijk
this code filters out the non-covid data that was used as test data in the a-proof project (and is thus also test data for the ICF classification task for this project) from an original csv containing all data from 2017. The a-proof project only used non-covid data from 2017. 
The testdata for the ICF task was filtered out of the pre-training data with this script in order to aviod any circularity.
Since the original data could not be provided due to privacy constraints, filepaths are hardcoded.
"""

import pickle
import pandas as pd

#read in test data from ICF experiment
path = "../../Non_covid_data_15oct/a-proof-exp/test_data_batch1_disregard_removed.pkl"

print("Reading pickle files...")
with open(path, "rb") as pkl_file:
    data = pickle.load(pkl_file)
 
 
#gather id's from the sentences used for testing batch 1
keys = []    
for instance in data:
    key = instance.key
    sen_id = instance.sen_id
    sen = instance.sen
    keys.append([key, sen_id, sen])

note_ids = []
for key in keys:
    #le = len(str(key[1]))
    if key[1] == '':
        note_ids.append(key[0].split('---')[1])
    #else:
    #    note_ids.append([key[0].split('---')[1][:-le], key[2]])

# create a set of the list so that you end up with 1 id per note instead of 1 id per sentence
columns_to_delete = list(set(note_ids))

# make column id's match with df
delete = []
for item in columns_to_delete:
    number = int(item)
    delete.append(number-1)
    
# read in csv with all data
original_csv = "/data/notes/vumc/all_data/notities_2017_deel2_cleaned.csv"

# initialize csv file where all data minus the test data will be gathered
outfile = "/data2/Documents/thesis_stella2/gather_traindata/notities_VUMC_2017_deel2_without_a-proof_testdata.csv"
df_original = pd.read_csv(original_csv, sep=',')

test1 = df_original['Unnamed: 0'].tolist()
test2 = df_original['notitieID'].tolist()

# select all data minus test data
df_selection = df_original.loc[df_original['Unnamed: 0'].isin(delete)==False]

#write to csv file
df_selection.to_csv(outfile, sep=';', index = False)
