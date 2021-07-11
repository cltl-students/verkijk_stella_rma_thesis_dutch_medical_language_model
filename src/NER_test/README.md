# Evaluating on Named Entity Recognition for general Dutch (out-of-domain extrinsic evaluation)
This is code to fine-tune language models on NER using simpletransformers (https://github.com/ThilinaRajapakse/simpletransformers)

# Data
The data used is from the CoNLL-2002 task for NER in Dutch (can be downloaded from here: https://www.clips.uantwerpen.be/conll2002/ner/). 
In the data folder presented here, the data is provided in text format, which is how it was feeded to the simpletransformers architecture for the tests presented in the thesis report.

# Script
To run the test, 4 arguments should be passed when running from commandline: modeltype, path_to_model, traindata, evaldata
Example of how to run to test BERTje:
python run_NER.py 'bert', 'GroNLP/bert-base-dutch-cased', 'data/ned_train_text.txt', 'data/ned_testb_text.txt'

