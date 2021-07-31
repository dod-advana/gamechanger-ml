# Entity Linking & Named Entity Recognition (NER)

## CoNLL Format
The Hugging Face format follows CoNLL-2003. Every token in a sentence (or sequence) is on a separate
line along with its "B-", "I-", "O" tag. In our corpus, we have GCPER (person) and GCORG 
(organization), with their abbreviations suffixed with "-ABBRV".

For example the sequence _The Director, DLA and the VA shall:_ would be represented as
```
The O
Director B-GCPER
, I-GCPER
DLA I-GCPER
and O
the O
DLA I-GCORG-ABBRV
shall O
: O

```
Sequences are separated by a single new line.

## Creating CoNLL Training Data
This is a multi-step process at the moment. We'll use DoDD, DoDI, and DoDM documents from the corpus as 
an example.

1. Create sentence `.csv` files. Run the CLI `src/text_classif/cli/raw_text2csv.py` with 
glob "DoD[DIM]*.json".This will write a `<doc-id>_sentences.csv` file, _e.g._, `DoDD_1000.20_sentences.csv` 
for each matching document, to a specified output directory.

2. We'll need a randomly chosen subset of the sentences, so first `cat` these files into one file, 
_e.g._, 
    ```
    cat *sentences.csv > your/output_path/big_sentence_file.csv
    ```
   
3. Next, shuffle and select *n* samples, _e.g._, 3,000 sentences:
    ```
    sort -R big_sentence_file.csv | head -3000 > rnd_3K_my_big_sentences.csv
    ```
   For the DoD[DIM] collection, this might take a few tens of seconds. If the file is very large,
   consider using `reservoir.py`.
   
4. `ner_training_data.py` uses the output of Step 3 to create `train`, `test`, and `val` datasets according
    the specified `--train-split`. The remainder will be evenly split into `test` and `val`:
    ```
    python ner_training_data.py \
        --sentence-csv rnd_3K_big_sentence_file.csv \
        --entity-csv path_to/gamechanger-ml/gamechangerml/src/entity/aux_data/flat_entities.csv \
        --separator space \
        --n-samples 0 \
        --train-split 0.80 \
        --min-tokens 4 \
        --max-tokens 100
    ```
   This will create three files, `train.txt.tmp`, `test.txt.tmp`, and `val.txt.tmp` (.80, .10, .10) in CoNLL format.
   
   **NB**: Due to the tokenizing and tagging, the resulting files get very large, very quickly. For 3,000 sentences (430KB `.csv`),
   resulting `train.txt.tmp` is close to 80MB.

## Training
The shell script `entity/bin/sample_run.sh`, run from `entity/bin`, sets up a small test using the
data in `tests/test_data`. 

It first runs `preprocessing.py` on each of the three files. This insures the number of tokens in 
the model-tokenized sentence is less than a maximum tokens (128). If not, it 
splits the sentence. This can be a slow process. If you know the sentences are within your limit or 
you're willing to tolerate some truncation, this step can be skipped. Simply change the file extension 
from `.txt.tmp` to `.txt`.

By default, the trained model will be saved in the data directory, `tests/test_data/model`. The `model`
directory will be created.

During training, you may see the warnings
```
/opt/conda/envs/gc-venv-blue/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:64: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
```
and
```
[WARNING|training_args.py:423] 2021-07-29 20:59:05,363 >> Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future version. Using `--per_device_train_batch_size` is preferred.
```
These are harmless and are fixed in later versions of `torch`.
