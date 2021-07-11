import pandas as pd


def make_note_df(note_ids, labels):

    data = {'note_id': note_ids,
                'labels': labels}
    df = pd.DataFrame(data)
    return(df)



def noteLabels(df):

    all_labels = []
    labels = []
    ids = []
    all_ids = []
    i = 0
    for index, row in df.iterrows():
        #append labels and note id's 
        labels.append(df.iloc[i]['labels'])
        ids.append(df.iloc[i]['note_id'])
        try:
            #see if note id changes in df
            if df.iloc[i]['note_id'] != df.iloc[i+1]['note_id']:
                #if so, append the list in which you collected labels and note id's earlier to a bigger list
                all_labels.append(labels)
                all_ids.append(ids)
                #and empty the lists in which you collect seperate labels and note id's
                labels = []
                ids = []
            i += 1
        except IndexError:
            #make sure you can append the last list as well
            all_labels.append(labels)
            all_ids.append(ids)
            labels = []
            ids = []

    labels_per_note = []
    for entry in all_labels:
        #eliminate double labels
        s = set(entry)
        l = list(s)
        labels_per_note.append(l)
        
    unique_ids = []
    for entry in all_ids:
        i_d = entry[0]
        unique_ids.append(i_d)
        

    final_list = []
    for l in labels_per_note:
        if len(l) > 1:
            #remove 'None' labels in notes where there is a domain label
            l.remove('None')
            final_list.append(l)
        else:
            final_list.append(l)
            
    return(final_list, unique_ids)





        










