"""
Tests a transformer language model on the CoNLL-2002 NERC task for Dutch
4 arguments should be passed when running from commandline: modeltype, path_to_model, traindata, evaldata
Example of how to run to test BERTje:
python run_NER.py 'bert', 'GroNLP/bert-base-dutch-cased', 'data/ned_train_text.txt', 'data/ned_testb_text.txt'
"""

import simpletransformers
import os
import sys
from simpletransformers.ner import NERModel, NERArgs

def run_ner(modeltype, path_to_model, traindata, evaldata)
    model_args = NERArgs()
    model_args.labels_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    model_args.classification_report = True

    model = NERModel(
        modeltype,
        path_to_model,
        args=model_args,
    )

    model.train_model(traindata)
    result, model_outputs, wrong_preds = model.eval_model(evaldata)

    return(result)

modeltype = sys.argv[1]
path_to_model = sys.argv[2]
traindata = sys.argv[3]
evaldata = sys.argv[4]
