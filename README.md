# GC - Machine Learning
## Table of Contents
1. [Directory](##Directory)
2. [Development Rules](#Development-Rules)
3. [Train Models](#Train-Models)
4. [ML API](#ML-API)

## Directory
```
├── gamechangerml
│   ├── api
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── docker-compose.override.yml
│   │   ├── docker-compose.yml
│   │   ├── fastapi
│   │   ├── getInitModels.py
│   │   ├── kube
│   │   ├── logs
│   │   ├── tests
│   │   └── utils
│   ├── configs
│   ├── corpus
│   ├── data
│   │   ├── features
│   │   │   ├── abbcounts.json
│   │   │   ├── abbreviations.csv
│   │   │   ├── abbreviations.json
│   │   │   ├── agencies.csv
│   │   │   ├── classifier_entities.csv
│   │   │   ├── combined_entities.csv
│   │   │   ├── corpus_doctypes.csv
│   │   │   ├── enwiki_vocab_min200.txt
│   │   │   ├── generated_files
│   │   │   │   ├── __init__.py
│   │   │   │   ├── common_orgs.csv
│   │   │   │   ├── corpus_meta.csv
│   │   │   │   └── prod_test_data.csv
│   │   │   ├── popular_documents.csv
│   │   │   ├── topics_wiki.csv
│   │   │   └── word-freq-corpus-20201101.txt
│   │   ├── ltr
│   │   ├── nltk_data
│   │   ├── test_data
│   │   ├── training
│   │   │   └── sent_transformer
│   │   ├── user_data
│   │   │   ├── gold_standard.csv
│   │   │   ├── matamo_feedback
│   │   │   │   ├── Feedback.csv
│   │   │   │   └── matamo_feedback.csv
│   │   │   └── search_history
│   │   │       └── SearchPdfMapping.csv
│   │   └── validation
│   │       ├── domain
│   │       │   ├── query_expansion
│   │       │   ├── question_answer
│   │       │   └── sent_transformer
│   │       └── original
│   │           ├── msmarco_1k
│   │           ├── multinli_1.0
│   │           └── squad2.0
│   ├── mlflow
│   ├── models
│   │   ├── ltr
│   │   ├── msmarco_index
│   │   ├── qexp_20211001
│   │   ├── sent_index_20211108
│   │   ├── topic_models
│   │   └── transformers
│   │       ├── bert-base-cased-squad2
│   │       ├── distilbart-mnli-12-3
│   │       ├── msmarco-distilbert-base-v2
│   ├── scripts
│   ├── src
│   │   ├── featurization
│   │   │   ├── abbreviation.py
│   │   │   ├── abbreviations_utils.py
│   │   │   ├── extract_improvement
│   │   │   ├── generated_fts.py
│   │   │   ├── keywords
│   │   │   ├── make_meta.py
│   │   │   ├── rank_features
│   │   │   ├── ref_list.py
│   │   │   ├── ref_utils.py
│   │   │   ├── responsibilities.py
│   │   │   ├── summary.py
│   │   │   ├── table.py
│   │   │   ├── term_extract
│   │   │   ├── test_hf_ner.py
│   │   │   ├── tests
│   │   │   ├── topic_modeling.py
│   │   │   └── word_sim.py
│   │   ├── model_testing
│   │   ├── search
│   │   │   ├── QA
│   │   │   ├── embed_reader
│   │   │   ├── query_expansion
│   │   │   ├── ranking
│   │   │   ├── semantic
│   │   │   └── sent_transformer
│   │   ├── text_classif
│   │   ├── text_handling
│   │   └── utilities
│   ├── stresstest
│   ├── train
```

## Development Rules
- Everything in gamechangerml/src should be independent of things outside of that structure (should not need to import from dataPipeline, common, etc).
- Where ever possible, code should be modular and broken down into smallest logical pieces and placed in the most logical subfolder.
- Include README.md file and/or example scripts demonstrating the functionality of your code.
- Models/large files should not be stored on Github.
- Data should not be stored on Github, there is a script in the `gamechangerml/scripts` folder to download a corpus from s3.
- File paths in gamechangerml/configs config files should be relative to gamechangerml and only used for local testing purposes (feel free to change on your local machine, but do not commit to repo with system specific paths).
- A config should not be required as an input parameter to a function; however a config can be used to provide parameters to a function (`foo(path=Config.path)`, rather than `foo(Config)`).
- If a config is used for a piece of code (such as training a model), the config should be placed in the relevant section of the repo (dataPipeline, api, etc.) and should clearly designate which environment the config is for (if relevant).

## Getting Started
### To use gamechangerml as a python module
- `pip install .`
- you should now be able to import gamechangerml anywhere python is available.


## Train Models
1. Setup your environment, and make any changes to configs: 
- `source ./gamechangerml/setup_env.sh DEV`
2. Ensure your AWS enviroment is setup (you have a default profile)
3. Get dependencies
- `source ./gamechangerml/scripts/download_dependencies.sh`
4. For query expansion:
- `python -m gamechangerml.train.scripts.run_train_models --flag {MODEL_NAME_SUFFIX} --saveremote {True or False} --model_dest {FILE_PATH_MODEL_OUTPUT} --corpus {CORPUS_DIR}`
5. For sentence embeddings:
- `python -m gamechangerml.train.scripts.create_embeddings -c {CORPUS LOCATION} --gpu True --em msmarco-distilbert-base-v2`

## ML API
1. Setup your environment, make any changes to configs: 
- `source ./gamechangerml/setup_env.sh DEV`
2. Ensure your AWS enviroment is setup (you have a default profile)
3. Dependencies will be automatically downloaded and extracted.
4. `cd gamechangerml/api`
5. `docker-compose build`
6. `docker-compose up`
7. visit `localhost:5000/docs`

## HELPFUL FLAGS FOR API
- export CONTAINER_RELOAD=True to reload the container on code changes for development
- export DOWNLOAD_DEP=True to get models and other deps from s3
- export MODEL_LOAD=False to not load models on API start (only for development needs) 

## FAQ
- I get an error with redis on API start
  - export ENV_TYPE=DEV
- Do I need to train models to use the API?
  - No, you can use the pretrained models within the dependencies. 
- The API is crashing when trying to load the models.
  - Likely your machine does not have enough resources (RAM or CPU) to load all models. Try to exclude models from the model folder.
- Do I need a machine with a GPU?
  - No, but it will make training or inferring faster.
- What if I can't download the dependencies since I am external?
  - We are working on making models publically available. However you can use download pretrained transformers from HuggingFace to include in the models/transformers directory, which will enable you to use some functionality of the API. Without any models, there is still functionality available like text extraction avaiable. 

## PULL REQUESTS
*Please provide:*
1. Description - what is the purpose, what are the different features added i.e. bugfix, added upload capability to model, model improving
2. Reviewer Test - how to test it manually and if it is on a dev/test server. (if applicable) 
 ` i.e. hit post endpoint /search with payload {"query": "military"}`
3. Unit/Integration tests - screenshot or copy output of unit tests from GC_ML_TESTS_119, any other tests or metrics applicable
