#Evaluating on ICF classification as a medical downstream task (in-domain extrinsic evaluation)
This code serves to fine-tune and test models on classifying sentences and documents as being about one of four domains fromt the World Health Organisation's International Classification of Functioning, Disability and Health (WHO ICF).

##Data
The data for these tests can not be provided due to privacy issues.
Therefore, the example given underneath on how to run the main script contains example paths to the data.

##Scripts
The main script is finetune_on_ICF_with_stf.py
All other scripts support this script. 

finetune_on_ICF_with_stf.py finetunes a model on the ICF data using simpletransformers (https://github.com/ThilinaRajapakse/simpletransformers), and:
-saves the finetuned model
-saves the predictions of the finetuned model on the test set
-prints the results on sentence level
-writes the results on note level to a csv
Arguments should be given in the commandline in the right order:
python finetune_on_ICF_with_stf.py modeltype path_to_model epochs model_output_dir path_to_traindata path_to_testdata outfile_predictions path_to_note_results
an example run is:
python finetune_on_ICF_with_stf.py 'roberta' '../models/from_scratch' 1 '../models/finetuned_ICF/from_scratchh_ICF' '../data/df_tr_nc.csv' '../data/df_te_nc.csv' 'predictions/from_scratch_ICF.csv' 'results/from_scratch_ICF_note_level.csv'
