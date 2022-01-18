# Testing a language model's anonymity
To test the level of anonymity of the From Scratch Medical Model, a test set was created.
Because most names in the pre-training data were replaced with PERSON with spaCy, the model has a strong semantic representation for PERSON as an individual's name.
To create a test set, sentences that contained the PERSON token were collected and replaced by "[MASK]". It was then tested how often the model would predict a name in the place of the masked token performing the fill-mask task.
  
# Data
Because of privacy contraints, not all data used for this task can not be provided. Only the test set that contains sentences that were not included in the pre-training of MedRoBERTa.nl can be found under the name "anon_specific_testset_eval_public.csv". 
  
# Scripts
_gather_persons.py_ creates a dataset of 8000 sentences for the fill-mask task. From this dataset, 100 relevant sentences were selected for the test set. 
This was done twice, once with data used in the pre-training phase and once with unseen data.
_anonymizing_test.py_ gathers predictions of the From Scratch Language Model (40 predictions per sentence) and writes them to a csv file where
- the first row is a prediction made
- the second row is how many times the prediction was made
  
