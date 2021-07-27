#!/bin/bash
#--------------------------------------------------------------------
# This sets up everything for NER model training
# Adapted from Hugging Face run.sh
#--------------------------------------------------------------------

# !!! TEST ONLY - UPDATE FOR YOUR ENVIRONMENT !!!
export ML_ROOT=../../../../
echo "content root: $ML_ROOT"

export NER_DIR=$ML_ROOT/gamechangerml/src/entity

# path to train, test, dev data   !!! TEST ONLY - UPDATE FOR YOUR ENVIRONMENT !!!
export DATA_DIR=$NER_DIR/tests/test_data

# parameters for the tokenizer & model
export BERT_MODEL=distilbert-base-uncased
export MAX_LENGTH=128
export BATCH_SIZE=8
export NUM_EPOCHS=1
export SAVE_STEPS=750
export SEED=1

export PYTHONPATH=$ML_ROOT:$PYTHONPATH

if ! [ -f $DATA_DIR/labels.txt ]; then
  echo "finding unique labels..."
  cat $DATA_DIR/train.txt.tmp $DATA_DIR/val.txt.tmp $DATA_DIR/test.txt.tmp | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
fi

if ! [ -f $DATA_DIR/test.txt ]; then
  echo "preprocessing test.txt.tmp"
  python3 $NER_DIR/preprocess.py -d $DATA_DIR/test.txt.tmp -m $BERT_MODEL -l $MAX_LENGTH > $DATA_DIR/test.txt
fi

if ! [ -f $DATA_DIR/dev.txt ]; then
  echo "preprocessing dev.txt.tmp"
  python3 $NER_DIR/preprocess.py -d $DATA_DIR/val.txt.tmp -m $BERT_MODEL -l $MAX_LENGTH > $DATA_DIR/dev.txt
fi

if ! [ -f $DATA_DIR/train.txt ]; then
  echo "preprocessing train.txt.tmp"
  python3 $NER_DIR/preprocess.py -d $DATA_DIR/train.txt.tmp -m $BERT_MODEL -l $MAX_LENGTH > $DATA_DIR/train.txt
fi

if ! [ -f $DATA_DIR/train.txt ]; then
  echo "cannot find train.txt - make sure this path is correct: $DATA_DIR"
  exit 1
fi

python3 $NER_DIR/run_ner.py \
--task_type NER \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $DATA_DIR/model \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
