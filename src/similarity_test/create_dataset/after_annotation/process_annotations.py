"""
@Author StellaVerkijk
takes three csv's: each have the same sentences but are annotated by a different person
because of privacy issues the seperate annotated csv's cannot be provided
the three csv's are checked against each other to make sure all names are replaced by PERSON
the three csv's are processed so that only sentences where all three annotators annotated the same are written to a new csv
this csv becomes the final csv of the annotator team. Later on, csv's of all annotator teams were merged together to create the final similarity test set of 824 triples.
"""

dir_csv1 = "../Annotations/annotatie-team2.1_stella.csv"
dir_csv2 = "../Annotations/annotatie-team2.2_caroline.csv"
dir_csv3 = "../Annotations/annotatie-team2.3_quinten_jane_micky.csv"

import pandas as pd

annot1 = pd.read_csv(dir_csv1, sep = ';', encoding = 'utf-8')
annot2 = pd.read_csv(dir_csv2, sep = ';', encoding = 'utf-8')
annot3 = pd.read_csv(dir_csv3, sep = ';', encoding = 'utf-8')

# Make sure all anonymization is done
# take annot1 as final result 

#sen1
for index, row in annot2.iterrows():
    if 'PERSON' in annot2.iloc[index]['Sen1'] and 'PERSON' not in annot1.iloc[index]['Sen1']:
        annot1.at[index, 'Sen1'] = annot2.iloc[index]['Sen1']
        print(annot1.iloc[index]['Sen1'])
        print(annot2.iloc[index]['Sen1'])
#sen2       
for index, row in annot2.iterrows():
    if 'PERSON' in annot2.iloc[index]['Sen2'] and 'PERSON' not in annot1.iloc[index]['Sen2']:
        annot1.at[index, 'Sen2'] = annot2.iloc[index]['Sen2']
        print(annot1.iloc[index]['Sen2'])
        print(annot2.iloc[index]['Sen2'])
#sen3      
for index, row in annot2.iterrows():
    if 'PERSON' in annot2.iloc[index]['Sen3'] and 'PERSON' not in annot1.iloc[index]['Sen3']:
        annot1.at[index, 'Sen3'] = annot2.iloc[index]['Sen3']
        print(annot1.iloc[index]['Sen3'])
        print(annot2.iloc[index]['Sen3'])
        
#sen1
for index, row in annot3.iterrows():
    if 'PERSON' in annot3.iloc[index]['Sen1'] and 'PERSON' not in annot1.iloc[index]['Sen1']:
        annot1.at[index, 'Sen1'] = annot3.iloc[index]['Sen1']
        print(annot1.iloc[index]['Sen1'])
        print(annot3.iloc[index]['Sen1'])
#sen2       
for index, row in annot3.iterrows():
    if 'PERSON' in annot3.iloc[index]['Sen2'] and 'PERSON' not in annot1.iloc[index]['Sen2']:
        annot1.at[index, 'Sen2'] = annot3.iloc[index]['Sen2']
        print(annot1.iloc[index]['Sen2'])
        print(annot3.iloc[index]['Sen2'])
#sen3      
for index, row in annot3.iterrows():
    if 'PERSON' in annot3.iloc[index]['Sen3'] and 'PERSON' not in annot1.iloc[index]['Sen3']:
        annot1.at[index, 'Sen3'] = annot3.iloc[index]['Sen3']
        print(annot1.iloc[index]['Sen3'])
        print(annot3.iloc[index]['Sen3'])

# ONLY SELECT ANNOTATIONS ALL THREE ANNOTATORS AGREED ON

annotations1 = annot1['Annotation']
annotations2 = annot2['Annotation']
annotations3 = annot3['Annotation']
all_annotations = list(zip(annotations1, annotations2, annotations3))
print(len(all_annotations)) #should be the same length as the original df

indices = []
i = -1
for tup in all_annotations:
    i +=1
    if '/' not in tup: 
        if '?' not in tup:
            if int(tup[0]) == int(tup[1]) == int(tup[2]):
                if 0 not in tup:
                    if '0' not in tup:
                        indices.append(i)
                        print(tup)
        
print(len(indices)) # should be shorter than length of original dfs

ids = []
for number in indices:
    ids.append(annot1.iloc[number]['IDs'])
    
print(len(ids)) # should be the same length as 'indices'

# take doubly anonimized df and filter
new_df = annot1[(annot1['IDs'].isin(ids))]
print(new_df)
df = new_df.drop(columns = 'Comments')
print(df)
outdir = ("agreed_annotations_team2.csv")

df.to_csv(outdir, sep = ';', encoding = 'utf-8')
