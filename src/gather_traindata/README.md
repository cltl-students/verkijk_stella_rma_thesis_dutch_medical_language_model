# Collecting pre-training data
...

# Data
The data can not be provided due to privacy constraints

# Scripts
The data for this project was provided by the AMC and VuMC in csv's. These csv's contained several types of information apart from the actual note (the text), such as the patient ID, the type of note.
Before processing the text in the hospital notes, some data needed to be filtered out.
The folder filter_out_unwanted_data contains two scripts:
  - _filter_batch1_test_notes_ filters out any data that was used as test data in the ICF downstream task
  - _filter_covid_notes_ was used to filter out any data about covid patients used in the a-proof project, in case we want to test the medical models on the ICF test for covid data in the future.
 
get_txt_data ....
