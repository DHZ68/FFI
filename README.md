# FFI
This repo is the Python 3 implementation of FFI (Anonymous Link).


## Table of Contents
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Data](#Data)
- [Code](#Code)
 

## Introduction
This project aims at __Euphemism Identification__


## Requirements
The code is based on Python 3.7. Please install the dependencies as below:  
```
pip install -r requirements.txt
```


## Data
Due to the license issue, we will not distribute the dataset ourselves, but we will direct the readers to their respective sources.  

__Drug__/__Weapon__/__Sexuality__: 
- _Raw Text Corpus_: Please request the dataset from [Self-Supervised Euphemism Detection and Identification for Content Moderation (Zhu et al. 2021)]
- _Ground Truth_: Please request the dataset from [Self-Supervised Euphemism Detection and Identification for Content Moderation (Zhu et al. 2021)]


__Sample__:
- _Raw Text Corpus_: we take the sample dataset `data/sample.txt` from the public available data from [Self-Supervised Euphemism Detection and Identification for Content Moderation (Zhu et al. 2021)] for the readers to run the code.
- _Ground Truth_: same as the Drug dataset (see `data/euphemism_answer_drug.txt` and `data/target_keywords_drug.txt`), these dataset are also public available from [Self-Supervised Euphemism Detection and Identification for Content Moderation (Zhu et al. 2021)].  
- This Sample dataset is only for you to play with the code and it does not represent any reliable results. 


## Code
### 1. Fine-tune the BERT model. 
Please refer to [this link from Hugging Face](https://huggingface.co/bert-base-uncased/) to fine-tune a BERT on a raw text corpus.

### 2. Euphemism Identification (For fair comparison, our euphemism detection procedure is consistent with (Zhu et al. 2021))
```
python ./Main.py --dataset sample --target drug  
```
You may find other tunable arguments --- `c1`, `c2` and `coarse` to specify different classifiers for euphemism identification. 
Please go to `Main.py` to find out their meanings. 


