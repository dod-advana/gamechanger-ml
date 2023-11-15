# gamechanger-ml/gamechangerml/experimental

## query_expansion_example.py

**Script Notes:** This is commandline application (CLI) that demonstrates query expansion. You will
need to build an index in order to run this example. Details on how that is accomplished can be
found in `src/search/query_expansion/README.md`.

Usage for this script is

```
usage: query_expansion_example.py [-h] -i INDEX_DIR -q QUERY_FILE

python query_expansion_example.py

optional arguments:
  -h, --help            show this help message and exit
  -i INDEX_DIR, --index-dir INDEX_DIR
                        ANN index directory
  -q QUERY_FILE, --query-file QUERY_FILE
                        text file containing one sample query per line
```

## entity_extraction_example.py

**Script Notes:** NB Experimental. This is a commandline application (CLI) demonstrating named
entity recognition (NER) using either the [spaCy](https://spacy.io/usage/linguistic-features#named-entities) NER or
the [Hugging Face transformers](https://github.com/huggingface/transformers) NER. Usage for this script
is

```
usage: entity_extraction_example.py [-h] -m {spacy,hf} [-c CORPUS_DIR]

Example Named Entity Extraction (NER)

optional arguments:
  -h, --help            show this help message and exit
  -m {spacy,hf}, --method {spacy,hf}
  -c CORPUS_DIR, --corpus-dir CORPUS_DIR
                        corpus directory
```

Entities of type ORG (organization) and LAW (legal) are extracted using spaCy. `hf` extracts type
ORG entities.
