# GC - Machine Learning
## Table of Contents
1. [Directory](##Directory)
2. [Development Rules](#Development-Rules)
3. [Train Models](#Train-Models)
4. [ML API](#ML-API)
5. [Helpful Flags For API](#Helpful-Flags-For-API)
6. [FAQ](#FAQ)
7. [Pull Requests](#Pull-Requests)


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
- Everything in `gamechangerml/src` should be independent of things outside of that structure (should not need to import from dataPipeline, common, etc).


### Configs
- Config files go in `gamechangerml/configs`. When you add a new class, import it in [gamechangerml/configs/__init__.py](gamechangerml/configs/__init__.py).
- File paths in `gamechangerml/configs/*` should be relative to `gamechangerml` and only used for local testing purposes. Feel free to change on your local machine, but ***do not commit system specific paths to the repository***.
- A config class (i.e., from `gamechangerml/configs/*`) should not be required as an input parameter to a function. However, a config class attribute can be used to provide parameters to a function (`foo(path=Config.path)`, rather than `foo(Config)`).


### What Can Be Stored On GitHub?
- Models and large files should *NOT* be stored on Github.
- Data should *NOT* be stored on Github, there is a script in the `gamechangerml/scripts` folder to download a corpus from s3.

### Use Best Practices
- Code should be modular, broken down into smallest logical pieces, and placed in the most logical subfolder.
- All classes, functions, etc. should have clear, concise, and consistent docstrings. 
  - Function docstrings should include:
    - A short description 
    - Any important remarks
    - Parameter types, defaults, and descriptions
    - Return types and descriptions

    Example:
    ```python
    def say(words, loud=False):
      """Make the animal say words.

      Args:
        words (str): Words for the animal to say.
        loud (bool): True to make the animal say the words loudly, False to 
          make the animal say the words in a normal tone. Default is False.

      Returns:
        None
      """
    ```
- Include a maximum of 1 class per file.
- Include README.md files that contain what, why, and how code is used.


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

## Helpful Flags For API
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

## Pull Requests
*Please provide:*
1. Description - what is the purpose, what are the different features added i.e. bugfix, added upload capability to model, model improving
2. Reviewer Test - how to test it manually and if it is on a dev/test server. (if applicable) 
 ` i.e. hit post endpoint /search with payload {"query": "military"}`
3. Unit/Integration tests - screenshot or copy output of unit tests from GC_ML_TESTS_119, any other tests or metrics applicable
