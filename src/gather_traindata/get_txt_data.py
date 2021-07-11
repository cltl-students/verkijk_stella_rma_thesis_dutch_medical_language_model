"""
@Author StellaVerkijk
Gather pre-training data for the creation of the domain-specific medical language models.
It loads csv's, adapts the row that contains the note by anonymizing it and dividing it in chunks and then exports it to a .txt file.
This script contains a function for each hospital & year.
"""

import pandas as pd
import re
import glob
import time
import spacy
import numpy as np
nlp = spacy.load('nl_core_news_lg')

def process_AMC_2018():

    #CODE FOR AMC VALRISICO DATA 2018
    start_time=time.time()
    lens = []
    note_count = 0
    for path in glob.glob("/data/notes/amc/*2018_part?.csv"):
        df = pd.read_csv(path, header = 0, index_col = None, sep = ',', encoding = 'utf-8')
        with open ('data/anonymised/final/notes_2018_AMC_valrisico_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
            for index, row in df.iterrows():
                doc = row['notitie']
                note_count+=1
                if note_count == 500000:
                    print("amount of time taken for 50.000 notes = ", time.time() - start_time)
                if note_count == 1000000:
                    print("amount of time taken for 100.000 notes = ", time.time() - start_time)
                if note_count == 1500000:
                    print("amount of time taken for 150.000 notes = ", time.time() - start_time)
                if note_count == 2000000:
                    print("amount of time taken for 200.000 notes = ", time.time() - start_time)
                try:
                    spacy_doc = nlp(doc)
                    anonymised = str(spacy_doc)
                    for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                        if e.label_ == 'PERSON':
                            anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                        if e.label_ == 'GPE':
                            anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    spacy_anon = nlp(anonymised)
                    sentences = []
                    for sentence in spacy_anon.sents:
                        sentences.append(sentence)
                    #chunks = np.array_split(sentences, 6)
                    #chunks = [sentences[i:i + 60] for i in range(0, len(sentences), 6)]
                    n = 40
                    chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                    lens_per_doc = []
                    for chunk in chunks:
                        lens_per_chunk = []
                        for sentence in chunk:
                            lens_per_chunk.append(len(sentence))
                            if str(sentence).endswith('.'):
                                outfile.write(str(sentence)+(' '))
                            else:
                                outfile.write(str(sentence))
                        outfile.write('\n')
                        lens_per_doc.append(lens_per_chunk)
                    lens.append(lens_per_doc)
                except:
                    print("Doc could not be read into spacy")
                
    list_sums = []
    print(len(lens))
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("AMC 2018 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def process_AMC_2017():
    
    #CODE FOR AMC VALRISICO DATA 2017
    start_time=time.time()
    lens = []
    note_count = 0
    for path in glob.glob("/data/notes/amc/*2017_part?.csv"):
        df = pd.read_csv(path, header = 0, index_col = None, sep = ',', encoding = 'utf-8')
        with open ('data/anonymised/final/notes_2017_AMC_valrisico_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
            for index, row in df.iterrows():
                note_count+=1
                doc= row['notitie']
                if note_count == 500000:
                    print("amount of time taken for 50.000 notes = ", time.time() - start_time)
                if note_count == 1000000:
                    print("amount of time taken for 100.000 notes = ", time.time() - start_time)
                if note_count == 1500000:
                    print("amount of time taken for 150.000 notes = ", time.time() - start_time)
                if note_count == 2000000:
                    print("amount of time taken for 200.000 notes = ", time.time() - start_time)
                try:
                    spacy_doc = nlp(doc)
                    anonymised = str(spacy_doc)
                    for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                        if e.label_ == 'PERSON':
                            anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                        if e.label_ == 'GPE':
                            anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    spacy_anon = nlp(anonymised)
                    sentences = []
                    for sentence in spacy_anon.sents:
                        sentences.append(sentence)
                    n = 40
                    chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                    lens_per_doc = []
                    for chunk in chunks:
                        lens_per_chunk = []
                        for sentence in chunk:
                            lens_per_chunk.append(len(sentence))
                            if str(sentence).endswith('.'):
                                outfile.write(str(sentence)+(' '))
                            else:
                                outfile.write(str(sentence))
                        outfile.write('\n')
                        lens_per_doc.append(lens_per_chunk)
                    lens.append(lens_per_doc)
                except:
                    print("Doc could not be read into spacy")
                
    list_sums = []
    print(len(lens))
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("AMC 2017 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def process_AMC_2020():
    
    # CODE FOR AMC 2020 data NO COVID
    lens = []
    note_count = 0
    start_time=time.time()
    path = "notities_AMC_2020_without_covid.csv"
    df = pd.read_csv(path, header = None, index_col = None, sep = ';', encoding = 'utf-8')

    with open ('data/anonymised/final/notes_2020_AMC_nocovid_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
        for index, row in df.iterrows():
            note_count+=1
            doc = row[5]
            if note_count == 500000:
                print("amount of time taken for 50.000 notes = ", time.time() - start_time)
            if note_count == 1000000:
                print("amount of time taken for 100.000 notes = ", time.time() - start_time)
            if note_count == 1500000:
                print("amount of time taken for 150.000 notes = ", time.time() - start_time)
            if note_count == 2000000:
                print("amount of time taken for 200.000 notes = ", time.time() - start_time)
            try:
                spacy_doc = nlp(doc)
                anonymised = str(spacy_doc)
                for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                    if e.label_ == 'PERSON':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    if e.label_ == 'GPE':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                spacy_anon = nlp(anonymised)
                sentences = []
                for sentence in spacy_anon.sents:
                    sentences.append(sentence)
                #chunks = np.array_split(sentences, 6)
                #chunks = [sentences[i:i + 60] for i in range(0, len(sentences), 6)]
                n = 40
                chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                lens_per_doc = []
                for chunk in chunks:
                    lens_per_chunk = []
                    for sentence in chunk:
                        lens_per_chunk.append(len(sentence))
                        if str(sentence).endswith('.'):
                            outfile.write(str(sentence)+(' '))
                        else:
                            outfile.write(str(sentence))
                    outfile.write('\n')
                    lens_per_doc.append(lens_per_chunk)
                lens.append(lens_per_doc)
            except:
                print("Doc could not be read into spacy")
                
    list_sums = []
    print('AMC 2020 data') 
    print(len(lens))
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("AMC 2018 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



def process_VUMC_2020():
    
    # CODE FOR VUMC 2020 data NO COVID
    lens = []
    note_count = 0
    start_time=time.time()
    path = "notities_VUMC_2020_without_covid.csv"
    df = pd.read_csv(path, header = None, index_col = None, sep = ';', encoding = 'utf-8')
    
    with open ('data/anonymised/final/notes_2020_VUMC_nocovid_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
        for index, row in df.iterrows():
            note_count+=1
            doc = row[5]
            if note_count == 500000:
                print("amount of time taken for 50.000 notes = ", time.time() - start_time)
            if note_count == 1000000:
                print("amount of time taken for 100.000 notes = ", time.time() - start_time)
            if note_count == 1500000:
                print("amount of time taken for 150.000 notes = ", time.time() - start_time)
            if note_count == 2000000:
                print("amount of time taken for 200.000 notes = ", time.time() - start_time)
            try:
                spacy_doc = nlp(doc)
                anonymised = str(spacy_doc)
                for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                    if e.label_ == 'PERSON':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    if e.label_ == 'GPE':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                spacy_anon = nlp(anonymised)
                sentences = []
                for sentence in spacy_anon.sents:
                    sentences.append(sentence)
                n = 40
                chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                lens_per_doc = []
                for chunk in chunks:
                    lens_per_chunk = []
                    for sentence in chunk:
                        lens_per_chunk.append(len(sentence))
                        if str(sentence).endswith('.'):
                            outfile.write(str(sentence)+(' '))
                        else:
                            outfile.write(str(sentence))
                    outfile.write('\n')
                    lens_per_doc.append(lens_per_chunk)
                lens.append(lens_per_doc)
            except:
                print("Doc could not be read into spacy")
    
    list_sums = []
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("VUMC 2020 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
        

def process_VUMC_2017_deel1():

    #CODE FOR VUMC valrisico data 2017 deel 1 
    lens = []
    note_count = 0
    start_time=time.time()
    path = "/data/notes/vumc/all_data/notities_2017_deel1_cleaned.csv"
    df = pd.read_csv(path, header = 0, index_col = None, sep = ',', encoding = 'utf-8')
    
    with open ('data/anonymised/final/notes_2017_VUMC_deel1_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
        for index, row in df.iterrows():
            note_count+=1
            doc = row['notitie']
            if note_count == 500000:
                print("amount of time taken for 50.000 notes = ", time.time() - start_time)
            if note_count == 1000000:
                print("amount of time taken for 100.000 notes = ", time.time() - start_time)
            if note_count == 1500000:
                print("amount of time taken for 150.000 notes = ", time.time() - start_time)
            if note_count == 2000000:
                print("amount of time taken for 200.000 notes = ", time.time() - start_time)
            try:
                spacy_doc = nlp(doc)
                anonymised = str(spacy_doc)
                for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                    if e.label_ == 'PERSON':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    if e.label_ == 'GPE':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                spacy_anon = nlp(anonymised)
                sentences = []
                for sentence in spacy_anon.sents:
                    sentences.append(sentence)
                #chunks = np.array_split(sentences, 6)
                #chunks = [sentences[i:i + 60] for i in range(0, len(sentences), 6)]
                n = 40
                chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                lens_per_doc = []
                for chunk in chunks:
                    lens_per_chunk = []
                    for sentence in chunk:
                        lens_per_chunk.append(len(sentence))
                        if str(sentence).endswith('.'):
                            outfile.write(str(sentence)+(' '))
                        else:
                            outfile.write(str(sentence))
                    outfile.write('\n')
                    lens_per_doc.append(lens_per_chunk)
                lens.append(lens_per_doc)
            except:
                print("Doc could not be read into spacy")
                
    list_sums = []
    print('VUMC 2017 data - deel 1')
    print(len(lens))
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("AMC 2018 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



def process_VUMC_2017_deel2():
    
    #CODE FOR VUMC valrisico data 2017 deel 2 without a-proof test data    
    lens = []
    note_count = 0
    start_time=time.time()
    path = "notities_VUMC_2017_deel2_without_a-proof_testdata.csv"
    df = pd.read_csv(path, header = 0, index_col = None, sep = ';', encoding = 'utf-8')
    
    with open('data/anonymised/final/notes_2017_VUMC_deel2_notestdata_anony_final.txt', 'a', encoding = 'utf-8') as outfile:
        for index, row in df.iterrows():
            note_count+=1
            doc = row['notitie']
            if note_count == 500000:
                print("amount of time taken for 50.000 notes = ", time.time() - start_time)
            if note_count == 1000000:
                print("amount of time taken for 100.000 notes = ", time.time() - start_time)
            if note_count == 1500000:
                print("amount of time taken for 150.000 notes = ", time.time() - start_time)
            if note_count == 2000000:
                print("amount of time taken for 200.000 notes = ", time.time() - start_time)
            try:
                spacy_doc = nlp(doc)
                anonymised = str(spacy_doc)
                for e in reversed(spacy_doc.ents): #reversed to not modify the offsets of other entities when substituting
                    if e.label_ == 'PERSON':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                    if e.label_ == 'GPE':
                        anonymised = anonymised[:e.start_char] + e.label_ + anonymised[e.end_char:]
                spacy_anon = nlp(anonymised)
                sentences = []
                for sentence in spacy_anon.sents:
                    sentences.append(sentence)
                #chunks = np.array_split(sentences, 6)
                #chunks = [sentences[i:i + 60] for i in range(0, len(sentences), 6)]
                n = 40
                chunks = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n -1) // n)]
                lens_per_doc = []
                for chunk in chunks:
                    lens_per_chunk = []
                    for sentence in chunk:
                        lens_per_chunk.append(len(sentence))
                        if str(sentence).endswith('.'):
                            outfile.write(str(sentence)+(' '))
                        else:
                            outfile.write(str(sentence))
                    outfile.write('\n')
                    lens_per_doc.append(lens_per_chunk)
                lens.append(lens_per_doc)
            except:
                print("Doc could not be read into spacy")
                
    list_sums = []
    print('VUMC 2017 data - deel 2')
    print(len(lens))
    for doc in lens:
        for chunk in doc:
            list_sums.append(sum(chunk))
    avg_length = sum(list_sums) / len(list_sums)
    print("Average length per chunk :", avg_length)
    i=0
    for item in list_sums:
        if item > 512:
            i += 1
    print("AMC 2018 data")
    print("Total amount of chunks: ", len(list_sums))
    print("Amount of chunks larger than 512: ", str(i))
    print("Largest length of chunk: ", max(list_sums))
    print("Amount of notes: ", note_count)
    hours,rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    

#process_AMC_2020()
#process_VUMC_2017_deel1()
process_VUMC_2017_deel2()




















































    



