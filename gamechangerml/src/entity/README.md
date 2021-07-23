# Entity Linking & Named Entity Recognition (NER)

## CoNLL Format
The Hugging Face format follows CoNLL-2003. Every token in a sentence (or sequence) is on a separate
line along with its "B-", "I-", "O" tag. In our corpus, we have GCPER (person) and GCORG (organization),
thus there are five unique tags.

For example the sentence _The Director, DLA and the VA shall:_ would be represented as
```
The O
Director B-GCPER
, I-GCPER
DLA I-GCPER
and O
the O
VA I-GCORG
shall O
: O
```
Sequences are separated by a single new line.

## Creating CoNLL Training Data
This is a multi-step process at the moment. We'll use DoDD, DoDI, and DoDM documents from the corpus.

1. Create sentence `.csv` files. Run the CLI `src/text_classif/cli/raw_text2csv.py` with glob "DoD[DIM]*.json".
This will write a `<doc-id>_sentences.csv`, _e.g._, `DoDD_1000.20_sentences.csv` for each
matching document, to a specified output directory.

2. We'll need to UAR sample a subset of the sentences, so first `cat` these files into one file, 
_e.g._, 
    ```
    cat *sentences.csv > your/output_path/big_sentence_file.csv
    ```
   
3. Next, shuffle and select *n* samples, _e.g._, 2,500 sentences:
    ```
    sort -R big_sentence_file.csv | head -3000 > rnd_3K_my_big_sentences.csv
    ```
   For the DoD[DIM] collection, this might take a few tens of seconds. If the file is very large,
   consider using `reservoir.py`.
   
4. `ner_training_data.py` uses the output of Step 3 to create `train`, `dev`, and `val` datasets
    ```
    python ner_training_data.py \
        --sentence-csv rnd_3K_big_sentence_file.csv \
        --entity-csv you_path_to/gamechanger-ml/gamechangerml/src/entity/aux_data/entities.csv \
        --separator space \
        --n-samples 0 \
        --train-split 0.80
    ```
   This will create three files, `train.txt.tmp`, `dev.txt.tmp`, and `val.txt.tmp` (.80, .10, .10). 
   Just prior to training the NER model, these files will pass through `preprocess.py` creating the required
   input files `train.txt`, `dev.txt`, and `val.txt`.
   
   **NB** Due to the tagging, etc., the resulting files get very large, very quickly. For 3,000 sentences (430KB `.csv`),
   the `train.txt.tmp` clocks in at 93MB.
   
 