# Pre-training the language models
This code can be run to reproduce the medical language models presented in the thesis report, or to produce any domain-specific language model, changing the training parameters.

## Data
The data for training the medical language models can not be provided due to privacy constraints.

## Scripts
There are two scripts, one for training a language model from scratch (from_scratch.py) and one for extending pre-training on RobBERT (extend_robbert.py).
In the thesis report, different phases of training are presented. The training_arguments.json file now contains an example of a trial run with the arguments used for the first phase of the From Scratch Model.
Anyone can change the values of the dictionary in the json file and run the script for a different phase or for a new experiment.
When running the scripts extend_robbert.py, one argument must be included in the command line: freeze_layers to only train the embedding-lookup layer (phase I) or not_freeze_layers (phase II and III)

The file adapted_robbert_class.py contains a custom made dataloader that is used for the training of the models. This is a LineByLineDataset class supported by PyTorch based on the dataloader of RobBERT.
