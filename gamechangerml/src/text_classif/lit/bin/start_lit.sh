#!/bin/bash

python nlp_lit/classifier_lit.py \
    --model_path ~/classifier_models/distilbert_3_label_dod_dim_b16_dropp_00_min_0_lr15_128_all_20210902_epoch_1 \
    --num_labels 3 \
    --data_path ~/classifier_data/dodd-sentences/DoDD_1020.1_CH_1_sentences.csv \
    --batch_size 16 \
    --max_seq_len 128 \
    --port 5432
