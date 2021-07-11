"""
@author StellaVerkijk
prints scores on the similarity test
"""

from simpletransformers.language_representation import RepresentationModel
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig, BertModel, BertTokenizer, BertConfig
import sklearn
from sklearn import metrics
from scipy import spatial
import pandas as pd
import numpy as np
import torch
from collections import Counter

def cosine_similarity_calc(vec_1,vec_2):
	
	sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
	
	return(sim)

def get_sen_reps(sentence, model, tokenizer):

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embeddings = last_hidden_states[0]
    
    sentence_embedding = torch.mean(embeddings, dim=0)
    
    return(sentence_embedding)

def get_odd_one_out(sentences, model, tokenizer):
    
    #print(vectors.shape) #should be (3, 768), one sentence embedding per sentence
    vectors = []
    vectors.append(get_sen_reps(sentences[0], model, tokenizer))
    vectors.append(get_sen_reps(sentences[1], model, tokenizer))
    vectors.append(get_sen_reps(sentences[2], model, tokenizer))
    
    #get similarity of each sentence compared to each sentence # both options give the same result for bertje # both options give equally diverting results per run for robbert
    sim12 = cosine_similarity_calc(vectors[0], vectors[1])
    sim23 = cosine_similarity_calc(vectors[1], vectors[2])
    sim13 = cosine_similarity_calc(vectors[0], vectors[2])

    #see which sentence has the lowest combined similarity score
    score_dict = dict()
    score_dict['Sen1'] = sim12 + sim13
    score_dict['Sen2'] = sim12 + sim23
    score_dict['Sen3'] = sim13 + sim23

    # find key with lowest value
    odd_one = min(score_dict, key=score_dict.get)
    return(odd_one)


def choose_model(modeltype):
    """
    Initializes model and tokenizer 
    :param modeltype: str ('robbert', 'bertje' 'mbert')
    """
    if modeltype == 'robbert':
        model = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base") #, config = config)
        tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        
    if modeltype == 'bertje':
        model = BertModel.from_pretrained("GroNLP/bert-base-dutch-cased") #, config=config)
        tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    if modeltype == 'mbert':
        model = BertModel.from_pretrained("bert-base-multilingual-cased") #, config=config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    return(model, tokenizer)


def get_score(modeltype):
    """
    Returns the score of the chosen model on the complete similarity test
    :param modeltype: str ('robbert', 'bertje' 'mbert')
    """
    
    model, tokenizer = choose_model(modeltype)
    #set path to test set --> change to 'complete_simtest_no_keywords_roberta.csv' or 'complete_simtest_no_keywords_bert.csv' to test on data set without keywords
    path_to_simtest = '../create_dataset/complete_simtest.csv'
    df_test = pd.read_csv(path_to_simtest, sep = ';')

    predictions = []
    for index, row in df_test.iterrows():
        sen1 = df_test.iloc[index]['Sen1']
        sen2 = df_test.iloc[index]['Sen2']
        sen3 = df_test.iloc[index]['Sen3']
        sentences = (sen1, sen2, sen3)
        odd_one = get_odd_one_out(sentences, model, tokenizer)
        predictions.append(odd_one)
        

    y_true = df_test['Annotation'].tolist()

    result = accuracy_score(y_true, predictions)
    return(result)
    

def get_score_per_triple_type(modeltype, triple_type):
    """
    Returns the score of the chosen model on the complete similarity test
    :param modeltype: str ('robbert', 'bertje' 'mbert')
    :param triple_type: int (1, 2, 3, 4)
    """
    
    model, tokenizer = choose_model(modeltype)
    #set path to test set --> change to 'complete_simtest_no_keywords_roberta.csv' or 'complete_simtest_no_keywords_bert.csv' to test on data set without keywords
    path_to_simtest = '../create_dataset/complete_simtest.csv'
    df_test = pd.read_csv(path_to_simtest, sep = ';')
    
    predictions = []
    annotations = []
    for index, row in df_test.iterrows():
            if 'v'+str(triple_type) in df_test.iloc[index]['IDs']:
                sen1 = df_test.iloc[index]['Sen1']
                sen2 = df_test.iloc[index]['Sen2']
                sen3 = df_test.iloc[index]['Sen3']
                sentences = (sen1, sen2, sen3)
                odd_one = get_odd_one_out(sentences, model, tokenizer)
                annotation = df_test.iloc[index]['Annotation']
                predictions.append(odd_one)
                annotations.append(annotation)
    
    result = accuracy_score(annotations, predictions)
    return(result)
            

result_complete = get_score('robbert')
print("Accuracy score on complete test set: ", result_complete)
result_t1 = get_score_per_triple_type('robbert', 1)
print("Accuracy score triple type 1: ", result_t1)
result_t2 = get_score_per_triple_type('robbert', 2)
print("Accuracy score triple type 2: ", result_t2)
result_t3 = get_score_per_triple_type('robbert', 3)
print("Accuracy score triple type 3: ", result_t3)
result_t4 = get_score_per_triple_type('robbert', 4)
print("Accuracy score triple type 4: ", result_t4)
