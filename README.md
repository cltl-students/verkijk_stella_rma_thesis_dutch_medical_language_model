# sample-thesis-project
This repository is an example for the structure and content that a CLTL thesis project may have. 

# Overview
This repository assumes a Python project, with an Open License (MIT style). If any of these aspects are different from your project please make sure to change those accordingly.
Please feel free to clone/fork this repository and use it as a template for your thesis.

# Project structure

```
thesis-project
└───data
│       │   sample_data.csv 
└───results
│       │   sample_results.png 
└───src
│   └───utils
│       │   plotting.py
│   │   main.py
│   .gitignore
│   LICENSE
│   README.md
│   requirements.tx
```

# To Do
Once you start, please go through the following steps to tailor this template to your project

## Thesis report
You may decide to upload your report here directly, or to simply add a reference to where the report is hosted (e.g. Overleaf)
- [ ] Add a reference to the thesis report

## Data 
To ensure reproducibility, Yu need to provide the data your project uses.
- [ ] Add your data in the data folder

Sometimes, sharing sharing data is not straightforward. For example, there may be restrictions regarding with whom or how you can share the data you have. Some other times, the data you are using is open and easily accessible from other sites, in which case you might want to point directly to the original source. Either way, if this is the case for you please 
- [ ] Add the data folder to ``.gitignore`` in order to avoid commiting these files to Github. For this you can simply uncomment the last line in the ``.gitignore`` file  
```
# Tailored ignored files
data/*
```
- [ ] Make sure to add a ``README.md`` file inside the data folder, where you explain in detail how to obtain and structure the data

## README
- [ ] Add instructions on how to set up the project, how to run your code and what to expect as an output.






