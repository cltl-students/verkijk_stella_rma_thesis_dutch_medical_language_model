# Evaluating on sentence similarity judgements (in-domain intrinsic evaluation)
The scripts in this folder contain 
- code for creating an odd-one-out similarity test set by selecting sentences from hospital notes using certain criteria.
- code for testing language models on the task of finding the odds one out per triple in this dataset.

## Data
The odd-one-out similarity test set contains sentences from Dutch hospital notes. Because all sentences were anonymized by at least three people, we received permission from the VUMC privacy officer to publish the final data set. The preliminary data sets that were created before annotation and from which samples were taken for annotation can not be provided.

## Scripts
The folder create_dataset contains two subfolders. 
In _before_annotation_, code is provided that was used to create a large sample of relevant triples to be annotated.
  - _get pre_annotated_sentences.py_ gathers sentences with ICF domain annotations (annotated during the a-proof project: https://github.com/cltl/a-proof).
  - _create_triples.py_ selects triples according to several specific criteria:
           1 = two sentences from same domain matched on  keyword + one sentence from same domain not matched on keyword
           2 = two sentences from same domain + one sentence from other domain
           3 = two sentences from same domain matched on keyword + one sentence from other domain
           4 = three sentences from same domain, one with different level
In _after_annotation_, code is provided that was used to process the annotated sets of triples
  - _process_annotaions.py_ checks three csv files of the same sentences annotated by three different people against each other and selects only those triples that have been annotated three times with the same label
  - _mask_keywords_ adapts the complete_simtest.csv file in the data folder so that all keywords used in the criteria mentioned above are replaced with the RoBERTa mask token.

The main file in this folder is _get_similarity_test_scores.py_.
This script prints scores on the similarity test (overall score over complete similarity data set as well as score per triple type) and can be run from the commandline providing two arguments:
-modeltype ('bertje' or 'robbert' or 'mbert')
-path to the dataset for the similarity test ('data/complete_simtest.csv' or 'data/complete_simtest_no_keywords_roberta.csv' or 'data/complete_simtest_no_keywords_bert.csv')
As the medical language models are not published yet due to privacy contraints, these models are not yet included in this code.
Example run for testing RobBERT:
python get_similarity_test_scores.py robbert 'data/complete_simtest.csv'
