"""
@Author StellaVerkijk
This scripts
-reads in sentences that were annotated with an ICF domain in the a-proof project (this data is not provided because of privacy issues)
-matches batches of three sentences together according to certain criteria:
   1 = two sentences from same domain matched on  keyword + one sentence from same domain not matched on keyword
   2 = two sentences from same domain + one sentence from other domain
   3 = two sentences from same domain matched on keyword + one sentence from other domain
   4 = three sentences from same domain, one with different level
-provides each example with a unique key
-takes samples of these batches and combines them into one sample
-writes a csv file with a shuffled version of the sample
-this csv file was used to annotate (the unannotated csv file is not provided because of provicy issues)
"""


from get_annotated_sentences import createDataframe, sen_per_domain, createDataframeLevel
import itertools
import random
from random import sample
import pandas as pd

#Define keywords

keywords_lopen = [" FAC0 ", " FAC1 ", " FAC2 ", " FAC3 ", " FAC4 ", " FAC5 ", " FAC 0 ", " FAC 1 ", " FAC 2 ", " FAC 3 ", " FAC 4 ", " FAC 5 ", " Transfer ", " transfer ", "mobiliteit", "Mobiliteit", " tillift ", " Tillift ", " rolstoel ", " Rolstoel ", " stoel ", " Stoel ", " bed ", " Bed ", " stapjes ", " Stapjes ", " stap ", " Stap ", " stappen ", " Stappen "]

keywords_stemming = ['emotioneel', 'Emotioneel', 'droevig', 'Droevig', 'verdrietig', 'Verdrietig', 'huilt', 'Huilt', 'huilen', 'Huilen','blij', 'Blij', 'tevreden', 'Tevreden', 'rustig', 'Rustig', 'onrustig', 'Onrustig', 'apatisch', 'Apatisch', 'verward', 'Verward', " modulerend affect ", " affect vlak ", " somber ", " niet blij ", " vrolijk "]

keywords_beroep = ['kantoor', 'Kantoor', 'bouw', 'Bouw', 'niet naar school', 'Niet naar school', ' les ', ' Les ']

keywords_inspanning = [' saturatie ', ' Saturatie ', ' saturatiedip ', ' Saturatiedip ', 'conditie', 'Conditie', 'snel vermoeid', 'Snel vermoeid', 'vermoeid', 'Vermoeid', 'uitgeput', 'Uitgeput', 'snel moe', 'Snel moe', ' saturatie dip ', ' sport ', ' Sport ']

#Define paths to raw data 
df_c = createDataframe("../../Covid_data_11nov/a-proof-exp/traindata_covidbatch.pkl")
df_nc = createDataframe("../../Non_covid_data_15oct/a-proof-exp/train_data_batch1_disregard_removed.pkl")

#Read in sentences annotated with a domain
sentences_lopen_nc= sen_per_domain(df_nc, 'lopen')
sentences_stemming_nc = sen_per_domain(df_nc, 'stemming')
sentences_beroep_nc = sen_per_domain(df_nc, 'beroep')
sentences_inspanning_nc = sen_per_domain(df_nc, 'inspanningstolerantie')

sentences_lopen_c= sen_per_domain(df_c, 'lopen')
sentences_stemming_c = sen_per_domain(df_c, 'stemming')
sentences_beroep_c = sen_per_domain(df_c, 'beroep')
sentences_inspanning_c = sen_per_domain(df_c, 'inspanningstolerantie')

#Create a list of tuples with a sentence and a unique key per domain
lopen_nc = []
i=0
for sentence in sentences_lopen_nc:
    i+=1
    lopen_nc.append((sentence, 'l_nc' + str(i)))
    
stemming_nc = []
i=0
for sentence in sentences_stemming_nc:
    i+=1
    stemming_nc.append((sentence, 's_nc' + str(i)))
    
beroep_nc = []
i=0
for sentence in sentences_beroep_nc:
    i+=1
    beroep_nc.append((sentence, 'b_nc' + str(i)))
    
inspanning_nc = []
i=0
for sentence in sentences_inspanning_nc:
    i+=1
    inspanning_nc.append((sentence, 'i_nc' + str(i)))
    
lopen_c = []
i=0
for sentence in sentences_lopen_c:
    i+=1
    lopen_c.append((sentence, 'l_c' + str(i)))
    
stemming_c = []
i=0
for sentence in sentences_stemming_nc:
    i+=1
    stemming_nc.append((sentence, 's_c' + str(i)))

beroep_c = []
i=0
for sentence in sentences_beroep_c:
    i+=1
    beroep_c.append((sentence, 'b_c' + str(i)))
    
inspanning_c = []
i=0
for sentence in sentences_lopen_nc:
    i+=1
    inspanning_c.append((sentence, 'i_c' + str(i)))
    

#Create duos of sentences from the same domain and provide each duo with a unique key that is a combination of the sentence's keys
print("Making combinations...")
combis_l = []
i = 0
for a, b in itertools.combinations(lopen_nc, 2):
    i+=1
    combis_l.append((a[0], b[0], (a[1] + '-' +b[1])))
i = 0
for a, b in itertools.combinations(lopen_c, 2):
    i+=1
    combis_l.append((a[0], b[0], (a[1] + '-' +b[1])))
    
combis_s = []
i = 0
for a, b in itertools.combinations(stemming_nc, 2):
    i+=1
    combis_s.append((a[0], b[0], (a[1] + '-' +b[1])))
i = 0
for a, b in itertools.combinations(stemming_c, 2):
    i+=1
    combis_s.append((a[0], b[0], (a[1] + '-' +b[1])))

combis_i = []
i = 0
for a, b in itertools.combinations(inspanning_nc, 2):
    i+=1
    combis_i.append((a[0], b[0], (a[1] + '-' +b[1])))
i = 0
for a, b in itertools.combinations(inspanning_c, 2):
    i+=1
    combis_i.append((a[0], b[0], (a[1] + '-' +b[1])))
    
combis_b = []
i = 0
for a, b in itertools.combinations(beroep_nc, 2):
    i+=1
    combis_b.append((a[0], b[0], (a[1] + '-' +b[1])))
i = 0
for a, b in itertools.combinations(beroep_c, 2):
    i+=1
    combis_b.append((a[0], b[0], (a[1] + '-' +b[1])))
    

#Create sample of triple type 1 and provide each triple with a unique key that is a combination of the sentence's keys

#example of how to create samples for triple type 1 based on the 'lopen' domain
#create empty list
l1 = []
#loop over keywords for the domain
for keyword in keywords_lopen:
    #pick a random sentence for the domain
    random_sen = random.choice(lopen_nc)
    #loop through duo's of the domain
    for tup in combis_l:
        #select those duo's with matching keywords that are not in the random sentence
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen:
            triple = [tup[0], tup[1], random_sen[0]]
            #shuffle the triple so that the order is not always the same
            random.shuffle(triple)
            #add a unique key to the triple including the triple type indicator
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v1')
            #gather examples in a list of tuples
            tupl = tuple(triple)
            l1.append(tupl)

s1 = []
for keyword in keywords_stemming:
    random_sen = random.choice(stemming_nc)
    for tup in combis_s:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v1')
            tupl = tuple(triple)
            s1.append(tupl)

            
i1 = []
for keyword in keywords_inspanning:
    random_sen = random.choice(inspanning_nc)
    for tup in combis_i:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v1')
            tupl = tuple(triple)
            i1.append(tupl)

            
b1 = []
for keyword in keywords_beroep:
    random_sen = random.choice(beroep_nc)
    for tup in combis_b:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v1')
            tupl = tuple(triple)
            b1.append(tupl)



#Create sample of triple type 2 and provide each triple with a unique key that is a combination of the sentence's keys

#define lists containing the lists of tuples with sentences and keys per domain
domains_no_l = [inspanning_nc, beroep_nc, stemming_nc]
domains_no_i = [lopen_nc, beroep_nc, stemming_nc]
domains_no_s = [inspanning_nc, beroep_nc, lopen_nc]
domains_no_b = [inspanning_nc, lopen_nc, stemming_nc]

#example of how to create samples for triple type 2 based on the 'lopen' domain
#create empty list
l2 = []
#loop over duos of 'lopen' sentences
for tup in combis_l:
    #randomly pick a domain that's not 'lopen'
    dom = random.choice(domains_no_l)
    #randomly pick a sentence from that domain
    random_sen = random.choice(dom)
    #create a triple with the duo and the randomly picked sentence
    triple = [tup[0], tup[1], random_sen[0]]
    #shuffle the triple so that they do not some in the same order everytime
    random.shuffle(triple)
    #add the unique key for this triple including triple type indicator at the end of the key
    triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v2')
    #gather the examples in a list of tuples
    l2.append(tuple(triple))
    

i2 = []
for tup in combis_i:
    dom = random.choice(domains_no_i)
    random_sen = random.choice(dom)
    triple = [tup[0], tup[1], random_sen[0]]
    random.shuffle(triple)
    triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v2')
    i2.append(tuple(triple))
    
s2 = []
for tup in combis_s:
    dom = random.choice(domains_no_s)
    random_sen = random.choice(dom)
    triple = [tup[0], tup[1], random_sen[0]]
    random.shuffle(triple)
    triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v2')
    s2.append(tuple(triple))
    
b2 = []
for tup in combis_b:
    dom = random.choice(domains_no_b)
    random_sen = random.choice(dom)
    triple = [tup[0], tup[1], random_sen[0]]
    random.shuffle(triple)
    triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v2')
    b2.append(tuple(triple))
    


#Create sample of triple type 3 and provide each triple with a unique key that is a combination of the sentence's keys

#example of how to create samples for triple type 3 based on the 'lopen' domain
#create empty list
l3 = []
#loop through keywords for the domain
for keyword in keywords_lopen:
    #pick a random sentence from a randomly chosen domain that is not lopen
    dom = random.choice(domains_no_l)
    random_sen = random.choice(dom)
    #loop over duo's of the lopen domain
    for tup in combis_l:
        #select those duo's that contain overlapping keywords that is not in the random sentence from a different domain
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen[0]:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v3')
            tupl = tuple(triple)
            l3.append(tupl)

s3 = []
for keyword in keywords_stemming:
    dom = random.choice(domains_no_s)
    random_sen = random.choice(dom)
    for tup in combis_s:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen[0]:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v3')
            tupl = tuple(triple)
            s3.append(tupl)

            
i3 = []
for keyword in keywords_inspanning:
    dom = random.choice(domains_no_i)
    random_sen = random.choice(dom)
    for tup in combis_i:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen[0]:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1]+ '-' + 'v3')
            tupl = tuple(triple)
            i3.append(tupl)

            
b3 = []
for keyword in keywords_beroep:
    dom = random.choice(domains_no_b)
    random_sen = random.choice(dom)
    for tup in combis_b:
        if keyword in tup[0] and keyword in tup[1] and keyword not in random_sen[0]:
            triple = [tup[0], tup[1], random_sen[0]]
            random.shuffle(triple)
            triple.append(tup[2] + '-' +random_sen[1] + '-' + 'v3')
            tupl = tuple(triple)
            b3.append(tupl)

#Create sample of triple type 4 and provide each triple with a unique key that is a combination of the sentence's keys

#create a dataframe with all sentences annotated for one specific domain and include the level that was annotated in the dataframe
df_lopen = createDataframeLevel("../../Non_covid_data_15oct/a-proof-exp/train_data_batch1_disregard_removed.pkl", 'lopen')
df_stemming = createDataframeLevel("../../Non_covid_data_15oct/a-proof-exp/train_data_batch1_disregard_removed.pkl", 'stemming')
df_ins = createDataframeLevel("../../Non_covid_data_15oct/a-proof-exp/train_data_batch1_disregard_removed.pkl", 'inspanningstolerantie')

print("Collecting sentences per level...")

#Gather all sentences from the same domain in a list according to which level was annotated
lopen_0 = []
lopen_1 = []
lopen_2 = []
lopen_3 = []
lopen_4 = []
lopen_5 = []

for index, item in enumerate(df_lopen['level']):
    if df_lopen.iloc[index]['level'] == 0:
        lopen_0.append(df_lopen.iloc[index]['sen'])
    if df_lopen.iloc[index]['level'] == 1:
        lopen_1.append(df_lopen.iloc[index]['sen'])
    if df_lopen.iloc[index]['level'] == 2:
        lopen_2.append(df_lopen.iloc[index]['sen'])
    if df_lopen.iloc[index]['level'] == 3:
        lopen_3.append(df_lopen.iloc[index]['sen'])
    if df_lopen.iloc[index]['level'] == 4:
        lopen_4.append(df_lopen.iloc[index]['sen'])
    if df_lopen.iloc[index]['level'] == 5:
        lopen_5.append(df_lopen.iloc[index]['sen'])
        

stemming_0 = []
stemming_1 = []
stemming_2 = []
stemming_3 = []
stemming_4 = []

for index, item in enumerate(df_stemming['level']):
    if df_stemming.iloc[index]['level'] == 0:
        stemming_0.append(df_stemming.iloc[index]['sen'])
    if df_stemming.iloc[index]['level'] == 1:
        stemming_1.append(df_stemming.iloc[index]['sen'])
    if df_stemming.iloc[index]['level'] == 2:
        stemming_2.append(df_stemming.iloc[index]['sen'])
    if df_stemming.iloc[index]['level'] == 3:
        stemming_3.append(df_stemming.iloc[index]['sen'])
    if df_stemming.iloc[index]['level'] == 4:
        stemming_4.append(df_stemming.iloc[index]['sen'])
            

ins_0 = []
ins_1 = []
ins_2 = []
ins_3 = []
ins_4 = []

for index, item in enumerate(df_ins['level']):
    if df_ins.iloc[index]['level'] == 0:
        ins_0.append(df_ins.iloc[index]['sen'])
    if df_ins.iloc[index]['level'] == 1:
        ins_1.append(df_ins.iloc[index]['sen'])
    if df_ins.iloc[index]['level'] == 2:
        ins_2.append(df_ins.iloc[index]['sen'])
    if df_ins.iloc[index]['level'] == 3:
        ins_3.append(df_ins.iloc[index]['sen'])
    if df_ins.iloc[index]['level'] == 4:
        ins_4.append(df_ins.iloc[index]['sen'])
        
        
#Create triples with sentences of the same domain but with one having a different level annotated
def get_examples(list1, list2, original_list, letter):
    
    list1_with_keys = []
    for sentence in list1:
        for sen in original_list:
            try:
                if sentence == sen[0]:
                    list1_with_keys.append((sentence, sen[1]))
            except:
                list1_with_keys.append((sentence, 'nokey'))
            
        
    list2_with_keys = []
    for sentence in list2:
        for sen in original_list:
            try:
                if sentence == sen[0]:
                    list2_with_keys.append((sentence, sen[1]))
            except:
                list2_with_keys.append((sentence, 'nokey'))
        

    examples = []
    for a, b in itertools.combinations(list1_with_keys, 2):
        odd = random.choice(list2_with_keys)
        l = [a[0], b[0], odd[0]]
        random.shuffle(l)
        l.append(a[1]+'-'+b[1]+'-'+odd[1]+'-v4')
        examples.append(tuple(l))
    
    
    return(examples)
                
  
print("Making triples for triple type 4...")
  
example_lopen_1 = get_examples(lopen_0, lopen_3, lopen_nc, 'l')
example_lopen_2 = get_examples(lopen_2, lopen_4, lopen_nc, 'l')
example_lopen_3 = get_examples(lopen_0, lopen_2, lopen_nc, 'l')
example_lopen_4 = get_examples(lopen_1, lopen_3, lopen_nc, 'l')
example_lopen_5 = get_examples(lopen_1, lopen_2, lopen_nc, 'l')
example_lopen_6 = get_examples(lopen_0, lopen_4, lopen_nc, 'l')
example_lopen_7 = get_examples(lopen_3, lopen_4, lopen_nc, 'l')
example_lopen_8 = get_examples(lopen_0, lopen_1, lopen_nc, 'l')
example_lopen_9 = get_examples(lopen_1, lopen_4, lopen_nc, 'l')

example_stemming_1 = get_examples(stemming_0, stemming_3, stemming_nc, 's')
example_stemming_2 = get_examples(stemming_2, stemming_4, stemming_nc, 's')
example_stemming_3 = get_examples(stemming_0, stemming_2, stemming_nc, 's')
example_stemming_4 = get_examples(stemming_1, stemming_3, stemming_nc, 's')
example_stemming_5 = get_examples(stemming_1, stemming_2, stemming_nc, 's')
example_stemming_6 = get_examples(stemming_0, stemming_4, stemming_nc, 's')
example_stemming_7 = get_examples(stemming_3, stemming_4, stemming_nc, 's')
example_stemming_8 = get_examples(stemming_0, stemming_1, stemming_nc, 's')
example_stemming_9 = get_examples(stemming_1, stemming_4, stemming_nc, 's')

example_ins_1 = get_examples(ins_0, ins_3, inspanning_nc, 'i')
example_ins_2 = get_examples(ins_2, ins_4, inspanning_nc, 'i')
example_ins_3 = get_examples(ins_0, ins_2, inspanning_nc, 'i')
example_ins_4 = get_examples(ins_1, ins_3, inspanning_nc, 'i')
example_ins_5 = get_examples(ins_1, ins_2, inspanning_nc, 'i')
example_ins_6 = get_examples(ins_0, ins_4, inspanning_nc, 'i')
example_ins_7 = get_examples(ins_3, ins_4, inspanning_nc, 'i')
example_ins_8 = get_examples(ins_0, ins_1, inspanning_nc, 'i')
example_ins_9 = get_examples(ins_1, ins_4, inspanning_nc, 'i')

#Taking samples of eacg triple type

random_l1 = sample(l1, 326)
random_s1 = sample(s1, 326)
random_i1 = sample(i1, 326)
random_b1 = sample(b1, 22)
batch1 = random_l1 + random_s1 + random_i1 + random_b1

random_l2 = sample(l2, 234)
random_s2 = sample(s2, 233)
random_i2 = sample(i2, 233)
random_b2 = sample(b2, 300)
batch2= random_l2 + random_s2 + random_i2 + random_b2

random_l3 = sample(l3, 326)
random_s3 = sample(s3, 326)
random_i3 = sample(i3, 326)
random_b3 = sample(b3, 22)
batch3 = random_l3 + random_s3 + random_i3 + random_b3


random_v_l1 = sample(example_lopen_1, 38)
random_v_l2 = sample(example_lopen_2, 37)
random_v_l3 = sample(example_lopen_3, 37)
random_v_l4 = sample(example_lopen_4, 37)
random_v_l5 = sample(example_lopen_5, 37)
random_v_l6 = sample(example_lopen_6, 37)
random_v_l7 = sample(example_lopen_7, 37)
random_v_l8 = sample(example_lopen_8, 37)
random_v_l9 = sample(example_lopen_9, 37)

random_v_s1 = sample(example_stemming_1, 37)
random_v_s2 = sample(example_stemming_2, 37)
random_v_s3 = sample(example_stemming_3, 37)
random_v_s4 = sample(example_stemming_4, 37)
random_v_s5 = sample(example_stemming_5, 37)
random_v_s6 = sample(example_stemming_6, 37)
random_v_s7 = sample(example_stemming_7, 37)
random_v_s8 = sample(example_stemming_8, 37)
random_v_s9 = sample(example_stemming_9, 37)

random_v_i1 = sample(example_ins_1, 37)
random_v_i2 = sample(example_ins_2, 37)
random_v_i3 = sample(example_ins_3, 37)
random_v_i4 = sample(example_ins_4, 37)
random_v_i5 = sample(example_ins_5, 37)
random_v_i6 = sample(example_ins_6, 37)
random_v_i7 = sample(example_ins_7, 37)
random_v_i8 = sample(example_ins_8, 37)
random_v_i9 = sample(example_ins_9, 37)

batch4 = random_v_l1 + random_v_l2 + random_v_l3 + random_v_l4 + random_v_l5 + random_v_l6 + random_v_l7 + random_v_l8 + random_v_l9 + random_v_s1 + random_v_s2 + random_v_s3 + random_v_s4 + random_v_s5 + random_v_s6 + random_v_s7 + random_v_s8 + random_v_s9 + random_v_i1 + random_v_i2 + random_v_i3 + random_v_i4 + random_v_i5 + random_v_i6 + random_v_i7 + random_v_i8 + random_v_i9

#Merge all triple types together and shuffle 
final = batch1 + batch2 + batch3 + batch4
random.shuffle(final)

#Create dataframe of shuffled examples
df_odd_one_out = pd.DataFrame(final, columns = ['Sen1', 'Sen2', 'Sen3', 'ID'])
df_odd_one_out.to_csv('examples_testsets/simple_similarity/odd_one_out_bigger_4000.csv', index = False, sep = ';', encoding = 'utf-8')
