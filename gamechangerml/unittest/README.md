# gamechangerml/unittest

This directory contains unittests for `gamechangerml`.

## Directory Structure

```
gamechangerml/unittest
├──__init__.py
├──README.md
├──.gitkeep
├──sent_transformer_tests               Tests for the Sentence Transformer model
|    ├──test_sentence_encoder.py        Tests for SentenceEncoder class
|    ├──test_sentence_searcher.py       Tests for SentenceSearcher class
|    └──test_similarity_ranker.py       Tests for SimilarityRanker class
├──utils                                Utilities for tests
|    ├──__init__.py
|    └──verify_attribute.py             Helper functions for verifying object attributes
```

## Prerequisites

1. Complete the **Prerequisites** and **Installation** sections within [_Use gamechangerml as a Python Module_ in the main README](../../README.md#use-gamechangerml-as-a-python-module).
2. Spin up the API. See directions (number 1 through 6) in the [_ML API_ section of the main README](../../README.md#ml-api).

## Usage

1. Complete all steps in the [Prerequisites](#prerequisites) section.
2. Activate the virtual environment you created during the [Prerequisites](#prerequisites).
3. `cd` into your local `gamechanger-ml` repository.
4. Run the command:
   ```
   python gamechangerml/unittest/<test file path>
   ```
   - Replace \<test file name\> with the remaining path to the test file you want to run (e.g., `sent_transformer/test_sentence_encoder.py`).
5. Results will print to the terminal.
