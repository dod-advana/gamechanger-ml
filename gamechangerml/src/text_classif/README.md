# `text_classif` 

This is a framework of sorts for transfer learning using Hugging Face pre-trained models for sequence
classification. The overarching goal is to predict statements of responsibility contained in
DoDD, DoDI, and DoDM documents. After the prediction step is completed, an entity (organization or
person) is linked to each predicted responsibility statement and a `.csv` is produced with all
manner of ancillary data.

## Creating Training Data

Producing the aforementioned `.csv` depends on a trained model which in turn, depends on several data files and the corpus. 
All the required data for training and producing this table (excluding the corpus) is collected in a single directory
`classifier_data/`. This directory is archived on Advana Teams and is up-to-date as of 
this writing (see General > Data Science > Classifier Model). I create training data on my local machine
and transfer it to a GPU instance for training the model. 

**Step 1**: An `entity_mentions.json` is required. An update-to-date version is in `classifier_data/entity_mentions/`. 
To create this file, you'll need `classifier_data/entity_mentions/flat_entities_custom.csv`.
*This file needs to re-created only if the entities change. Its validity for other document types is not known*.
```
python ~/gamechanger-ml/gamechangerml/src/entity/entity_mentions.py \
    --input-path your-path-to/gc-corpus \
    --entity-file your-path-to/classifier_data/entity_mentions/flat_entities_custom.csv \
    --glob DoD[DIM]*.json \
    --output-path your-path-to/classifier_data/entity_mentions/dod_dim_entity_mentions_2021MMdd.json \
    --task mentions
```

**Step 2**: Create the training data for targeted document types. The CLI is `gamechangerml/src/text_classif/cli/resp_training_text.py`.
I recommend running this on individually on each of the types DoDD, DoDI, and DoDM with the output directed to
the proper subdirectory of `classifier_data/`. This way, if anything goes wrong, previous work is not lost. 
For example, to create training data for DoDI documents:
```
python ~/gamechanger-ml/gamechangerml/src/text_classif/cli/resp_training_text.py \
    --input-dir your-path-to/gc-corpus \
    --glob DoDI*.json \
    --agencies-file your-path-to/classifier_data/agency/agencies.csv \
    --output your-path-to/classifier_data/3-label-training/train_dodi_3_label_dropp_00_min_0_20210MMdd.csv \
    --entity-csv your-path-to/classifier_data/entity_mentions/flat_entities_custom.csv \
    --drop-zero-prob 0.0 \
    --min-tokens 0
```
Update appropriately for DoDD and DoDM document types.

**Step 3**: Create final training data. For brevity, call the files of Step 2 `dodd_train.csv`, 
`dodi_train.csv`, and `dodm_train.csv`. Concatenate these individual files into one file
```
cat dodd_train.csv dodi_train.csv dodm_train.csv > ~/classifier_data/3-label-training/dod_dim_3_label_dropp_00_min_0_2021MMDD_all.csv
```
I recommend extracting a small random sample, say 10,000 examples, to test
model training, e.g.,
```
sort -R ~/classifier_data/3-label-training/dod_dim_3_label_dropp_00_min_0_2021MMDD_all.csv | head -10000 > ~/classifier_data/3-label-training/dod_dim_3_label_test_10K.csv 
```

## Model Training
A current version of the model is archived on Advana Teams under General > Data Science > Classifier Model

### Copy the training data to the GPU instance
Create a `.tar.gz` for `classifier_data/`:
```
tar -czf classifier_data.tar.gz classifier_data/
```
Use `ssh` to establish a connection to the GPU instance, e.g.,
```
ssh -i uot.pem -L 5432:localhost:5432 your_id@10.192.45.39
```
Copy the `.tar.gz` to your home directory on the GPU instance
```
scp -i path-to/uot.pem classifier_data.tar.gz your_id@10.192.45.39:. 
```
On the GPU instance, expand the `.tar.gz`
```
tar -xvf classifier_data.tar.gz
```
All the training data is now in `~/classifier_data`.

### Training
I'll illustrate training using the 10K subset. The "final" model should be trained on the
the full set of training data. One epoch is sufficient for training.
```
python ~/gamechanger-ml/gamechangerml/src/text_classif/examples/example_gc_cli.py \
    --config-yaml bin/singleresp_distilbert_gc_config.yml \
    --data-file ~/classifier_data/3-label-training/dod_dim_3_label_test_10K.csv \
    --model-type distilbert \
    --num-samples 0 \
    --checkpoint-path test_distilbert_3_label_dod
```
Note: The trained model will be written to `test_distilbert_3_label_test_epoch_1`. If this
directory exists, an error will be generated.

You should see something close to
```
2021-09-08 16:02:06.212941: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[2021-09-08 16:02:14,625    INFO], [log_init.py:53 - initialize_logger()], log file : distilbert-gc-example-dodd.log
[2021-09-08 16:02:14,655    INFO], [classifier.py:263 - _tokenize_encode()], max seq length : 128
[2021-09-08 16:02:14,655    INFO], [classifier.py:270 - _tokenize_encode()], tokenizing, encoding...
[2021-09-08 16:02:24,754    INFO], [classifier.py:288 - _tokenize_encode()], done tokenizing, encoding...
[2021-09-08 16:02:24,755    INFO], [classifier.py:163 - train_test_ds()],     train samples : 9,000
[2021-09-08 16:02:24,755    INFO], [classifier.py:164 - train_test_ds()],       val samples : 1,000
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_transform.weight', 'vocab_transform.bias', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_projector.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[2021-09-08 16:02:26,990    INFO], [distilbert_classifier.py:35 - load_model_tokenizer()], model is loaded : distilbert-base-uncased
[2021-09-08 16:02:29,279    INFO], [classifier.py:211 - load_optimizer()], optimizer is loaded : AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-08
    lr: 1e-05
    weight_decay: 0.0
)
[2021-09-08 16:02:29,279    INFO], [classifier.py:187 - load_scheduler()], load scheduler, epochs :   1
[2021-09-08 16:02:29,279    INFO], [classifier.py:196 - load_scheduler()], scheduler is loaded
[2021-09-08 16:02:29,280    INFO], [config.py:179 - log_config()], ----------------------------------------------------------
[2021-09-08 16:02:29,280    INFO], [config.py:194 - log_config()],                class : DistilBertClassifier
[2021-09-08 16:02:29,280    INFO], [config.py:194 - log_config()],              version :  0.8.3
[2021-09-08 16:02:29,280    INFO], [config.py:194 - log_config()],               config : singleresp_distilbert_gc_config.yml
[2021-09-08 16:02:29,280    INFO], [config.py:191 - log_config()],           num labels :       3
[2021-09-08 16:02:29,280    INFO], [config.py:194 - log_config()],               log id : distilbert-gc-example-dodd.log
[2021-09-08 16:02:29,280    INFO], [config.py:194 - log_config()],           model name : distilbert-base-uncased
[2021-09-08 16:02:29,280    INFO], [config.py:191 - log_config()],               epochs :       1
[2021-09-08 16:02:29,280    INFO], [config.py:191 - log_config()],           batch size :      16
[2021-09-08 16:02:29,280    INFO], [config.py:191 - log_config()],         random state :   6,666
[2021-09-08 16:02:29,280    INFO], [config.py:188 - log_config()],      checkpoint path :   None
[2021-09-08 16:02:29,280    INFO], [config.py:188 - log_config()],     tensorboard path :   None
[2021-09-08 16:02:29,280    INFO], [config.py:197 - log_config()],                split :   9.0E-01
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],         warmup steps :       0
[2021-09-08 16:02:29,281    INFO], [config.py:197 - log_config()],                   lr :   1.0E-05
[2021-09-08 16:02:29,281    INFO], [config.py:197 - log_config()],         weight decay :   0.0E+00
[2021-09-08 16:02:29,281    INFO], [config.py:197 - log_config()],                  eps :   1.0E-08
[2021-09-08 16:02:29,281    INFO], [config.py:197 - log_config()],       clip grad norm :   1.0E+00
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],            drop last :       0
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],             truncate :       1
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],          max seq len :     128
[2021-09-08 16:02:29,281    INFO], [config.py:194 - log_config()],           model type : distilbert
[2021-09-08 16:02:29,281    INFO], [config.py:194 - log_config()],               device :   cuda
[2021-09-08 16:02:29,281    INFO], [config.py:194 - log_config()],        training data : dod_dim_3_label_test_10K.csv
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],     training samples :   9,000
[2021-09-08 16:02:29,281    INFO], [config.py:191 - log_config()],   validation samples :   1,000
[2021-09-08 16:02:29,281    INFO], [config.py:194 - log_config()],            optimizer :  AdamW
[2021-09-08 16:02:29,282    INFO], [config.py:191 - log_config()],          total steps :     563
[2021-09-08 16:02:29,282    INFO], [config.py:204 - log_config()], ----------------------------------------------------------
[2021-09-08 16:02:29,282    INFO], [classifier.py:303 - train()], into the breach...
[2021-09-08 16:02:29,282    INFO], [classifier.py:308 - train()],           ==================== Epoch   1 /   1 ====================
[2021-09-08 16:02:29,673    INFO], [classifier.py:351 - _train_batch()], 	batch     1 /   563 	loss : 1.117	elapsed : 0:00:00
[2021-09-08 16:02:30,002    INFO], [classifier.py:351 - _train_batch()], 	batch     2 /   563 	loss : 1.071	elapsed : 0:00:01
[2021-09-08 16:02:30,291    INFO], [classifier.py:351 - _train_batch()], 	batch     3 /   563 	loss : 1.049	elapsed : 0:00:01
[2021-09-08 16:02:30,826    INFO], [classifier.py:351 - _train_batch()], 	batch     5 /   563 	loss : 1.036	elapsed : 0:00:02
[2021-09-08 16:02:31,620    INFO], [classifier.py:351 - _train_batch()], 	batch     8 /   563 	loss : 1.008	elapsed : 0:00:02
[2021-09-08 16:02:32,945    INFO], [classifier.py:351 - _train_batch()], 	batch    13 /   563 	loss : 0.951	elapsed : 0:00:04
[2021-09-08 16:02:35,069    INFO], [classifier.py:351 - _train_batch()], 	batch    21 /   563 	loss : 0.903	elapsed : 0:00:06
[2021-09-08 16:02:38,539    INFO], [classifier.py:351 - _train_batch()], 	batch    34 /   563 	loss : 0.701	elapsed : 0:00:09
[2021-09-08 16:02:44,152    INFO], [classifier.py:351 - _train_batch()], 	batch    55 /   563 	loss : 0.398	elapsed : 0:00:15
[2021-09-08 16:02:53,244    INFO], [classifier.py:351 - _train_batch()], 	batch    89 /   563 	loss : 0.319	elapsed : 0:00:24
[2021-09-08 16:03:07,966    INFO], [classifier.py:351 - _train_batch()], 	batch   144 /   563 	loss : 0.431	elapsed : 0:00:39
[2021-09-08 16:03:31,880    INFO], [classifier.py:351 - _train_batch()], 	batch   233 /   563 	loss : 0.154	elapsed : 0:01:03
[2021-09-08 16:04:10,617    INFO], [classifier.py:351 - _train_batch()], 	batch   377 /   563 	loss : 0.289	elapsed : 0:01:41
[2021-09-08 16:05:00,675    INFO], [classifier.py:393 - _train_batch()], 	batch   563 /   563 	loss : 0.352	elapsed : 0:02:31
[2021-09-08 16:05:00,675    INFO], [classifier.py:319 - train()], avg training loss : 0.2889
[2021-09-08 16:05:00,675    INFO], [classifier.py:419 - _validate()], running validation
100%|█████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:04<00:00, 13.61it/s]
[2021-09-08 16:05:05,307    INFO], [classifier.py:453 - _validate()],    best loss : inf
[2021-09-08 16:05:05,318    INFO], [classifier.py:471 - _validate()],

              precision    recall  f1-score   support

           0      0.935     0.941     0.938       598
           1      0.916     0.921     0.919       381
           2      0.800     0.571     0.667        21

    accuracy                          0.926      1000
   macro avg      0.884     0.811     0.841      1000
weighted avg      0.925     0.926     0.925      1000

[2021-09-08 16:05:05,329    INFO], [classifier.py:472 - _validate()], confusion matrix

	         pred: 0  pred: 1  pred: 2
true: 0      563       32        3
true: 1       30      351        0
true: 2        9        0       12

[2021-09-08 16:05:05,329    INFO], [classifier.py:473 - _validate()], 	validation loss : 0.223
[2021-09-08 16:05:05,330    INFO], [classifier.py:474 - _validate()], 	            MCC : 0.850
[2021-09-08 16:05:05,330    INFO], [classifier.py:476 - _validate()], 	 accuracy score : 0.926
[2021-09-08 16:05:05,330    INFO], [classifier.py:477 - _validate()], 	validation time : 0:00:05
[2021-09-08 16:05:05,330    INFO], [checkpoint_handler.py:29 - write_checkpoint()], saving model with loss : 0.223
[2021-09-08 16:05:05,828    INFO], [checkpoint_handler.py:36 - write_checkpoint()], model file written to : test_distilbert_3_label_test_epoch_1
[2021-09-08 16:05:05,828    INFO], [classifier.py:324 - train()], training time : 0:02:37
```

**Please make a `model-card.md` for your model and copy it to the model's directory.**

## Predicting Responsibilities
After training the model on the *full training set*, the CLI `predict_table.py` produces the
final product, e.g.,

```
python ~/gamechanger-ml/gamechangerml/src/text_classif/cli/predict_table.py \
    --model-path distilbert_3_label_dod_dim_b16_dropp_00_min_0_lr15_128_all_20210902_epoch_1/ \
    --data-path ~/gc-corpus/ \
    --batch-size 128 \
    --max-seq-len 128 \
    --num-labels 3 \
    --glob DoD[DIM]*.json \
    --output-csv ~/classifier_data/predict-table-3-label-20210903-test.csv \
    --entity-mentions ~/classifier_data/entity_mentions/dod_dim_entity_mentions_20210901.json \
    --entity-csv ~/classifier_data/entity_mentions/flat_entities_custom.csv \
    --agencies-file ~/classifier_data/agency/agencies.csv
```

Upon executing this, you should see something close to
```
[2021-09-03 19:23:53,556    INFO], [predict_table.py:112 - predict_table()], into the breach...
[2021-09-03 19:23:53,556    INFO], [entity_link.py:56 - __init__()], EntityLink version 0.8.3
[2021-09-03 19:23:53,556    INFO], [entity_link.py:71 - __init__()],  max seq len : 128
[2021-09-03 19:23:53,556    INFO], [entity_link.py:72 - __init__()],   batch size : 128
[2021-09-03 19:23:53,556    INFO], [entity_link.py:73 - __init__()],   num labels :   3
[2021-09-03 19:23:53,556    INFO], [entity_link.py:74 - __init__()],        top k :   3
[2021-09-03 19:23:53,654    INFO], [predict_table.py:123 - predict_table()], into the breach...
[2021-09-03 19:23:53,655    INFO], [predictor.py:89 - __init__()], Predictor v0.8.3
[2021-09-03 19:23:55,856    INFO], [predictor.py:53 - _log_metadata()],       checkpoint time : 2021-09-02 17:45:52
[2021-09-03 19:23:55,857    INFO], [predictor.py:57 - _log_metadata()],       current version : 0.8.3
[2021-09-03 19:23:55,857    INFO], [predictor.py:58 - _log_metadata()],  created with version : 0.8.3
[2021-09-03 19:23:55,857    INFO], [predictor.py:59 - _log_metadata()],        training class : DistilBertClassifier
[2021-09-03 19:23:55,857    INFO], [predictor.py:60 - _log_metadata()],            base model : distilbert-base-uncased
[2021-09-03 19:23:55,857    INFO], [predictor.py:61 - _log_metadata()],            num labels : 3
[2021-09-03 19:23:55,857    INFO], [predictor.py:62 - _log_metadata()],                 epoch : 1
[2021-09-03 19:23:55,857    INFO], [predictor.py:63 - _log_metadata()],          avg val loss : 0.099
[2021-09-03 19:23:55,857    INFO], [predictor.py:64 - _log_metadata()],                   mcc : 0.935
[2021-09-03 19:23:58,190    INFO], [predictor.py:112 - __init__()], model loaded
predict: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1392/1392 [33:06<00:00,  1.43s/it]
[2021-09-03 19:57:11,431    INFO], [predict_table.py:134 - predict_table()], writing predict-table-3-label-20210903-test-labels-gt-1.csv
[2021-09-03 19:57:11,477    INFO], [predict_table.py:138 - predict_table()], building agencies for entries 71,064 entries
[2021-09-03 19:57:11,477    INFO], [predict_table.py:139 - predict_table()], please be patient...
[2021-09-03 19:57:11,566    INFO], [abbreviations_utils.py:65 - get_agencies()], building intermediate table, size : 71,064
[2021-09-03 19:57:44,251    INFO], [abbreviations_utils.py:76 - get_agencies()], intermediate table built : 0:00:33
[2021-09-03 19:57:44,251    INFO], [abbreviations_utils.py:78 - get_agencies()], attaching agencies...
[2021-09-03 20:11:54,192    INFO], [abbreviations_utils.py:96 - get_agencies()], agencies attached : 0:14:10
[2021-09-03 20:11:54,201    INFO], [predict_table.py:148 - predict_table()], getting references...
refs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 71064/71064 [01:42<00:00, 691.30it/s]
[2021-09-03 20:13:37,677    INFO], [predict_table.py:164 - predict_table()], final csv written to /home/cskiscim/classifier_data/predict-table-3-label-20210903-test.csv
[2021-09-03 20:13:37,677    INFO], [predict_table.py:169 - predict_table()], total time : 0:49:44
```

## Save Your Work
The trained model and final table should be `scp`'d to your local machine and archived to, at this
writing, MS Teams.

## Classifier Details
The `base_class` does most of the heavy lifting, however, it **must** be subclassed to provide a few
model and data loading methods. Implemented are models based on

- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased`

The Hugging Face models can be downloaded from their ['models' site](https://huggingface.co/models).

> If you do have a required model, the underlying library will attempt to 
> download the required files. These will be cached in `~/.cache`. 

## Configuration
The classifier is driven entirely by a configuration file such as the ones
in the `examples/` directory. 

The schema for the YAML configuration is shown below. 

**NB**: *All* the 
configuration items must be present. If `None` is allowed as a value, use the
`~` character in the YAML to indicate `None`. Strict type checking is enforced as well as checking
for obvious errors. 

|Name | Type(s) | Description|
|:--- |:--- | :--- |
|log_id|`str`|Name of the log file|
|model_name|`str`|A valid Hugging Face model name|
|epochs|`int`|Number of epochs for model training|
|batch_size|`int`| Batch size; should be a power of 2|
|random_state|`int`|Sets the random number seed; can be `None`|
|checkpoint_path|`(str, None)`| Where to write the model file.|
|tensorboard_path|`(str, type_none)`|Directory for `tensorboard`'s `fevents` files.|
|num_labels|`int`|The number of unique labels|
|split|`float`|A number in (0, 1)|
|warmup_steps|`(float, None)`|Applied to the scheduler|
|lr|`float`|Learning rate|
|weight_decay|`float`|See the [AdamW documentation](https://huggingface.co/transformers/main_classes/optimizer_schedules.html)|
|eps|`float`|Stopping criteria for the optimizer|
|clip_grad_norm|`(float, None)`|Limits the gradient to a function of this value|
|drop_last|`bool`|If `True`, drop the last batch if it is incomplete|
|truncate|`bool`|If `True`, truncate to the max length. Recommend `False`|
|max_seq_len|`(int, None)`|If `None`, this value is computed from the encoded text. Otherwise, this value is used|

The schema is in `utils/config.py`. Changes to the schema are encouraged.