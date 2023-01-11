# gamechangerml.src.search.semantic_search

## Background
The idea behind semantic search is to embed all entries in your corpus (in our 
case, paragraphs) into a vector space.

At search time, the query is embedded into the same vector space and the
closest embeddings from the corpus are found. These entries should have a
high semantic overlap with the query.

Searching a large corpus with millions of embeddings can be time-consuming
if exact nearest neighbor search is used. In that case, Approximate Nearest
Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller
fractions of similar embeddings. This index can be searched efficiently and
the embeddings with the highest similarity (the nearest neighbors) can be
retrieved within milliseconds, even if you have millions of vectors.

---

## Semantic Search in the GAMECHANGER Application

Within the 
[GAMECHANGER web](https://gitlab.advana.boozallencsn.com/advana/gamechanger/gamechanger-web-source)
application, semantic search is used via the [ML API](../../../api/fastapi/) 
`transSentenceSearch` endpoint. In addition, semantic search is the foundation
of [document comparison](../document_comparison/document_comparison.py) which is 
used via the `/documentCompare` endpoint.

---

## Directory Structure
```
- gamechangerml/src/search/semantic_search
    ├── __init__.py
    ├── semantic_search.py                              SemanticSearch class: the main class for this module.
    ├── utils
    │   ├── __init__.py
    │   └── local_corpus_tokenizer.py                   LocalCorpusTokenizer class: tokenize documents at the paragraph level to prepare for embeddings.
    ├── train
    │   ├── __init__.py
    │   ├── semantic_search_finetuner.py                SemanticSearchFinetuner class: Finetune the sentence transformer model used for semantic search
    │   └── semantic_search_training_data.py            SemanticSearchTrainingData class: Create training data for SemanticSearchFinetuner.
    └── tests                                           Tests for this module
        ├── __init__.py
        ├── conftest.py                                 pytest configurations
        ├── test_prepare_corpus_for_embedding.py        Test for SemanticSearch.prepare_corpus_for_embedding()
        ├── test_create_embeddings_index.py             Test for SemanticSearch.create_embeddings_index()
        ├── test_search.py                              Test for SemanticSearch.search()
        └── test_data                                   
            ├── prepare_corpus_for_embedding            Input test data for test_prepare_corpus_for_embedding.py
            │   └── *.json                              Multiple JSON files
            ├── create_embeddings_index                 
            │   └── input.json                          Input test data for test_create_embeddings_index.py
            └── test_index                              Sample index for test_search.py
                ├── config
                ├── data.csv
                ├── doc_ids.txt
                ├── embeddings
                └── embeddings.npy
```

---

## Configurations

Configurations are stored [here](../../../configs/semantic_search_config.py)

---

## Prerequisites
1. Source environment variables. From the root of the repository, run:
    ```
    source ./gamechangerml/setup_env.sh DEV
    ```

2. Create a virtual environment.  
    - Follow instructions [here](../../../../docs/VENV.md)  
    - From the root of the repository, run:
        ```
        pip install .
        ```
3. The sentence transformer model must exist in the 
[transformers](../../../models/transformers) directory, in a folder with the
model name.
    - For example, if the model name is *msmarco-distilbert-base-v2*, then the
    model files must be in 
    \<[transformers directory path](../../../models/transformers)>/msmarco-distilbert-base-v2.
    The default model name is 
    [SemanticSearchConfigs.BASE_MODEL](../../../configs/semantic_search_config.py).
    - To download our pre-trained model from S3:
        - Refresh your AWSAML token.
        - Run the [download_dependencies.sh](../../../scripts/data_transfer/download_dependencies_from_s3.sh) script. Note: you only need the
        transformers models, so you may comment out everything else to save time
        and storage.


---

## Example Usage

1. Complete the [Prerequisites](#prerequisites) if you haven't already.
2. Activate the virtual environment you created in step 1.


```python
from os.path import join
from logging import getLogger
from gamechangerml.configs import SemanticSearchConfig
from gamechangerml.src.search.semantic_search import SemanticSearch
from gamechangerml.api.fastapi.settings import LOCAL_TRANSFORMERS_DIR, SENT_INDEX_PATH

### Setup ###
model_path = join(LOCAL_TRANSFORMERS_DIR.value, SemanticSearchConfig.BASE_MODEL)
index_path = SENT_INDEX_PATH.value
logger = getLogger(__name__)
# Path to a directory with JSON files (output from gamechanger-data parsers).
# Or, None to use a test corpus of MSMarcoData.
data_directory = "../data" 

### Create an Embeddings Index ###
semantics = SemanticSearch(
    model_path=model_path,
    index_directory_path=index_path,
    load_index_from_file=False,  # don't load an index if you're going to create one
    logger=logger,
    use_gpu=False
)
corpus = semantics.prepare_corpus_for_embedding(data_directory)
semantics.create_embeddings_index(corpus)

### Search ###
results = semantics.search(
    query="Major Automated Information System",
    num_results=3,
    preprocess_query=True,
    threshold=SemanticSearchConfig.DEFAULT_THRESHOLD_ARG
)
```

---

## How to Test
The [tests](./tests) for this module use `pytest`.

1. Complete the [Prerequisites](#prerequisites) if you haven't already.
2. Activate the virtual environment you created in step 1.
3. Navigate to `gamechanger-ml/gamechangerml/src/search/semantic_search`
4. Run:
    ```
    pytest
    ```

---

## Model Training

Use the [finetune_sentence_retriever.py](../../../scripts/finetune_sentence_retriever.py) script.

---
