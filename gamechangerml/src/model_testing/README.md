# Model Testing

The model evaluation pipeline contains classes for testing the transformer models used in Gamechanger search for different tasks:

- Semantic Search (`msmarco-distilbert-base-v2`) - sourced from huggingface, finetuned on msmarco
- Similarity Ranking (`distilbart-mnli-12-3`) - sourced from huggingface, finetuned on nli
- Question Answering (`bert-base-cased-squad`) - sourced from huggingface, finetuned on SQuAD 2.0

## Using the pipeline:
1. Make sure you have the validation data saved in `gamechangerml/data/validation`. 
2. Check the `gamechangerml/configs/config.py` file for changes to the defaults
3. To test a single model, import the class into your script that matches the validation dataset you want to use. Each pipeline has the option of testing on their original fine-tuned data (for getting baseline expectation of the model) or in-domain data (with the exception of the similarity model for now.)
4. To test all of the models on all available validation datasets, run `python gamechangerml/scripts/run_evaluation.py`. There is an option to pass in a sample limit for QA and the similarity model (because SQuAD and NLI test datasets can take a while to test completely). 
5. Each pipeline will output a JSON results file in the model directory and a detailed csv with results in `gamechangerml/data/evaluation`

## Metadata & Metrics
Each JSON results file contains metadata and aggregated metrics for the test:
- User
- Date
- Model Name
- Validation Data
- Query Count: total number of queries tested (for Similarity Model, there is also a Pairs Count for the total number of sentence pairs tested)

## QA Metrics:
- exact match: % of queries that returned answers exactly matching one of the expected answers
- partial match: % of queries that returned answers partially matching one of the expected answers

## Retriever Metrics:
There isn't a 1:1 relationship for all queries to expected documents returned in our validation data. 
- in top 10: For each query, a hit in the top 10 = 1, no hit = 0, and the score for the query is the average. This is the average of all query scores
- any hits: if a query had any hits returned that were expected, this score is 1. The aggregate score is the average. 

## Similarity Metrics:
- all match: the proportion of sentence pairs that exactly match the expected rankings (this matters more for the NLI dataset which has clear entailment/neutral/contradicting labels for each pair).
- top match: the proportion of top ranked matches that were re-ranked to the top by the model. This is more important for GC because these are the results that are returned by intelligent search.


# Old Testing Script
How to use the testing script:

python3 model_test.py

It will pull down the latest models and corpus from s3 and assess the models against that, then delete the model and json files when it's done.

To use testing script on local models:
Make sure all the model files are in the current working directory. Then:

python3 model_test.py --local

This will only pull the corpus json files from s3 and only assess the models in the current directory.

To use verbosity:

python3 model_test.py --verbose

To use gold standard test (against the gold_standard.csv, its layout is query | expected results separated by ;):
python3 model_test.py --gold_standard

To use iteration (get recall @ k, it will get the top 5 to top 50 documents from inference iterating by 5):
python3 model_test.py --iterate


Versions:
0.1.0 --> first push of the testing script, exact-match query test
0.2.0 --> added gold standard query/results dataset
