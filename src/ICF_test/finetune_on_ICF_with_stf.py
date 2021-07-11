"""
finetune medical language model with simpletransformers
@author StellaVerkijk

This scripts finetunes a model on the ICF data and 
-saves the finetuned model
-saves the predictions of the finetuned model on the test set
-prints the results on sentence level
-writes the results on note level to a csv

Arguments should be given in the commandline in the right order:
python finetune_on_ICF_with_stf.py modeltype path_to_model epochs model_output_dir path_to_traindata path_to_testdata outfile_predictions path_to_note_results
an example run is:
python finetune_on_ICF_with_stf.py 'roberta' '../models/from_scratch' 1 '../models/finetuned_ICF/from_scratchh_ICF' '../data/df_tr_nc.csv' '../data/df_te_nc.csv' 'predictions/from_scratch_ICF.csv' 'results/from_scratch_ICF_note_level.csv'
"""

import pandas as pd
import pickle
import torch
import simpletransformers
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from preprocessing import prepareDataNC
from class_definitions import Annotation, BertContainer
from utils import lightweightDataframe, completeDataframe, filterDataframe
from eval_domain_agg import eval_per_domain
from domain_classification import make_note_df, noteLabels


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def model_setup(modeltype, model_path, epochs, output_dir):
    
    model_args = ClassificationArgs()
    model_args.num_train_epochs = epochs
    model_args.output_dir = output_dir
    
    model = ClassificationModel(
    modeltype, model_path, use_cuda=False, num_labels = 5, args=model_args
    )

    return(model)

def train(model_to_train, path_to_train_df):
    
    # load dataset
    init_train_df = pd.read_csv(path_to_train_df, delimiter = ';')
    
    # select relevant columns
    train_df = init_train_df[['sentence', 'label']]

    # turn classes into numerical classes
    train_df.loc[train_df['label'] == 'None', 'label'] = 0
    train_df.loc[train_df['label'] == '.D450: Lopen en zich verplaatsen', 'label'] = 1
    train_df.loc[train_df['label'] == '.B152: Stemming', 'label'] = 2
    train_df.loc[train_df['label'] == '.B455: Inspanningstolerantie', 'label'] = 3
    train_df.loc[train_df['label'] == '.D840-859: Beroep en werk', 'label'] = 4
    
    # rename columns so simpletransformers recognises them
    train_df.columns = ['text', 'labels']

    model_to_train.train_model(train_df)


def predict(trained_model, path_to_test_df):

    # load dataset
    init_test_df = pd.read_csv(path_to_test_df, delimiter = ';')

    # select relevant columns
    test_df = init_test_df[['sentence', 'label']]

    # turn classes into numerical classes
    test_df.loc[test_df['label'] == 'None', 'label'] = 0
    test_df.loc[test_df['label'] == '.D450: Lopen en zich verplaatsen', 'label'] = 1
    test_df.loc[test_df['label'] == '.B152: Stemming', 'label'] = 2
    test_df.loc[test_df['label'] == '.B455: Inspanningstolerantie', 'label'] = 3
    test_df.loc[test_df['label'] == '.D840-859: Beroep en werk', 'label'] = 4
    
    # rename columns so simpletransformers recognises them
    test_df.columns = ['text', 'labels']
    
    # get predictions
    texts = test_df['text'].tolist()
    predictions, raw_outputs = trained_model.predict(texts)
    test_df['predictions'] = predictions
    
    return(test_df)


def evaluate(df):
    
    y_true = df['labels'].tolist()
    y_pred = df['predictions'].tolist()
    report = classification_report(y_true, y_pred)

    return(report)

def main(modeltype, path_to_model, epochs, model_output_dir, path_to_traindata, path_to_testdata, outfile_predictions, path_to_note_results):

    print("Initialize model...")


    # select model (modeltype, path to model, number of epochs, output dir)
    # path to model must contain the tokenizer files as well, and no training state file (in case you are finetuning on a checkpoint) 
    model_to_train = model_setup(modeltype, path_to_model, epochs, model_output_dir)

    print("Start training...")
    # define path to traindata
    train(model_to_train, path_to_traindata)

    print("Start predicting...")
    trained_model = ClassificationModel(modeltype, model_output_dir, use_cuda = False)
    test_df_with_predictions = predict(trained_model, path_to_testdata)
    test_df_with_predictions.to_csv(outfile_predictions, index=False)

    print("Evaluating on sentence level...")
    report = evaluate(test_df_with_predictions)
    print(report)
    
    print("Evaluating on note level...")
    note_ids = prepareDataNC(completeDataframe)[2]
    
    #get classes as strings again
    test_df_with_predictions.loc[test_df_with_predictions['labels'] == 0, 'labels'] = 'None'
    test_df_with_predictions.loc[test_df_with_predictions['labels'] == 1, 'labels'] = '.D450: Lopen en zich verplaatsen'
    test_df_with_predictions.loc[test_df_with_predictions['labels'] == 2, 'labels'] = '.B152: Stemming'
    test_df_with_predictions.loc[test_df_with_predictions['labels'] == 3, 'labels'] = '.B455: Inspanningstolerantie'
    test_df_with_predictions.loc[test_df_with_predictions['labels'] == 4, 'labels'] = '.D840-859: Beroep en werk'
    
    test_df_with_predictions.loc[test_df_with_predictions['predictions'] == 0, 'predictions'] = 'None'
    test_df_with_predictions.loc[test_df_with_predictions['predictions'] == 1, 'predictions'] = '.D450: Lopen en zich verplaatsen'
    test_df_with_predictions.loc[test_df_with_predictions['predictions'] == 2, 'predictions'] = '.B152: Stemming'
    test_df_with_predictions.loc[test_df_with_predictions['predictions'] == 3, 'predictions'] = '.B455: Inspanningstolerantie'
    test_df_with_predictions.loc[test_df_with_predictions['predictions'] == 4, 'predictions'] = '.D840-859: Beroep en werk'
    
    #make dataframe of note id's and labels for annotations: change variables according to traindata you selected
    labels = test_df_with_predictions['labels'].tolist()
    print(labels)
    df_annotations = make_note_df(note_ids, labels)
    #get annotations per note 
    annotations_per_note, unique_ids_man = noteLabels(df_annotations)

    #make dataframe of note id's and labels for predictions
    predictions = test_df_with_predictions['predictions'].tolist()
    df_predictions = make_note_df(note_ids, predictions)
    #get predicted labels per note
    predictions_per_note, unique_ids_sys = noteLabels(df_predictions)

    #make dictionaries for evaluation
    dict_predict = dict(zip(unique_ids_sys, predictions_per_note))
    dict_ann = dict(zip(unique_ids_man, annotations_per_note))

    #Write eval per domain on note level to tsv file
    eval_per_domain(dict_predict, dict_ann, path_to_note_results)

modeltype = sys.argv[1]
path_to_model = sys.argv[2]
epochs = sys.argv[3]
model_output_dir = sys.argv[4]
path_to_traindata = sys.argv[5]
path_to_testdata = sys.argv[6]
outfile_predictions = sys.argv[7]
path_to_note_results = sys.argv[8]
    

main(modeltype, path_to_model, epochs, model_output_dir, path_to_traindata, path_to_testdata, outfile_predictions, path_to_note_results)
