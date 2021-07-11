"""
@Author Quirine Smit 
@Author StellaVerkijk
this code filters out covid data an original csv containing data from 2020. 
Run twice and change filenames to collect filtered data for VuMC as well as AMC.
"""

import pandas as pd

def select_ids(df_diagnoses, search_7=[]):

    MDN_ids = set()

    for query in search_7:
        temp_set = set(df_diagnoses.loc[df_diagnoses[7] == query][0])
        MDN_ids.update(temp_set)

    return MDN_ids

def main(hospital, outputfile):
    
    
    # Read files
    if hospital == "AMC":
        # AMC
        diagnoses_filepath = "/data/bestanden 2020/Diagnoses AMC 2020 sept.csv"
        notities_filepath_1 = "/data/bestanden 2020/Notities AMC 2020 Q1.csv"
        notities_filepath_2 = "/data/bestanden 2020/Notities AMC 2020 Q2.csv"
        notities_filepath_3 = "/data/bestanden 2020/Notities AMC 2020 Q3.csv"
        
    if hospital == "VUMC":
        #VUmc
        diagnoses_filepath = "/data/bestanden 2020/Diagnoses VUMC 2020 sept.csv"
        notities_filepath_1 = "/data/bestanden 2020/Notities VUMC 2020 Q1.csv"
        notities_filepath_2 = "/data/bestanden 2020/Notities VUMC 2020 Q2.csv"
        notities_filepath_3 = "/data/bestanden 2020/Notities VUMC 2020 Q3.csv"
    
    
    # Define patient id pat_id_column
    pat_id_column = 0
    # pat_id_column = 'Pat_id'
    
    # Read in files as pd.DataFrame types
    df_diagnoses = pd.read_csv(diagnoses_filepath, sep=';', header=None, encoding = 'utf-8')
    
    df_notities_1 = pd.read_csv(notities_filepath_1, sep=';', header=None, encoding = 'utf-8-sig', engine='python', error_bad_lines=False)
    df_notities_2 = pd.read_csv(notities_filepath_2, sep=';', header=None, encoding = 'utf-8-sig', engine='python', error_bad_lines=False)
    df_notities_3 = pd.read_csv(notities_filepath_3, sep=';', header=None, encoding = 'utf-8-sig', engine='python', error_bad_lines=False)
    # Join notes to one df
    df_notities = pd.concat([df_notities_1, df_notities_2, df_notities_3])
    

    # Search queries
    search_7 =  ["COVID-19, virus geïdentificeerd [U07.1]"]

    # MDN_ids is patient id
    MDN_ids = select_ids(df_diagnoses, search_7=search_7)

    # Create df with selected MDN ids
    df_selection = df_notities.loc[df_notities[pat_id_column].isin(MDN_ids)==False]
    
    
    # Print statements for counts
    print(search_7)
    print("Aantal patient ids in search", len(MDN_ids))
    print("Patient ids die ook in notities staan", len(MDN_ids & set(df_notities[pat_id_column])))
    print("Aantal notities als die notities eruit zijn", df_selection.shape[0])
    print("Gemiddeld aantal documenten per patient", df_selection.shape[0]/len(MDN_ids & set(df_notities[pat_id_column])))

    # Write to csv
    #df_selection.to_csv(outputfile, sep=';', header = None, index = False)
    return(df_selection)
    
output_filepath = 'notities_VUMC_2020_without_covid.csv'
df_selection = main("VUMC", output_filepath)
df_selection.to_csv(output_filepath, sep=';', header = None, index = False)


#sanity check:
#diagnoses_filepath = "/data/bestanden 2020/Diagnoses AMC 2020 sept.csv"
diagnoses_filepath = "/data/bestanden 2020/Diagnoses VUMC 2020 sept.csv"
df_diagnoses = pd.read_csv(diagnoses_filepath, sep=';', header=None, encoding = 'utf-8')
covid_ids = select_ids(df_diagnoses, search_7=["COVID-19, virus geïdentificeerd [U07.1]"])
my_ids = set(df_selection[0].tolist())
intersection = list(covid_ids & my_ids)
print(len(intersection)) #should be zero
print(intersection) #should be empty list
