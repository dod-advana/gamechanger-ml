---
language: 
- en
tags:
- text classification
license: MIT
datasets:
- dod_dim_3_label_dropp_00_min_0_20210902_all.csv
metrics:
- sklearn classification report
- matthews correlation coefficient
---

# distilbert_3_label_dod_dim_b16_dropp_00_min_0_lr15_128_all_20210902_epoch_1 

## Model description

This is a multi-class sentence classifier intended to discover statements
of responsibility in DoD Issuance documents. It is built from the
pre-trained model distilbert-base-uncased.

## Intended uses & limitations

#### How to use
The data is validation dataset `.csv` with columns "src", "label", "sentence". If
the validation label is not known, use `0`. 

```python
from gamechangerml.src.text_classif.utils import predict_glob

model_path = "distilbert_3_label_dod_dim_b16_dropp_00_min_0_lr15_128_all_20210902_epoch_1"
data_path = "/path_to/gc-corpus"
match_files = "DoDD*.json"

for predictions, file_name in predict_glob(
            model_path,
            data_path,
            match_files,
            max_seq_len=128,
            batch_size=8,
            num_labels=3,
        ):
    # do something with predictions
```
`predictions` is a list of dictionaries with keys 
`['top_class', 'prob', 'src', 'label', 'sentence']`

'top_class' is the predicted label and 'prob' is the classification probability. 
The remaining keys and values are from the `data_path` csv.

#### Limitations and bias

- The model was trained on a narrow domain. It's general applicability is not known. 


## Training data
45,470 sentences from DoDD, DoDI, and DoDM documents

## Eval Results

loss: 0.1638
```
              precision    recall  f1-score   support

           0      0.974     0.976     0.975      2753
           1      0.963     0.966     0.964      1699
           2      0.893     0.789     0.838        95

    accuracy                          0.968      4547
   macro avg      0.943     0.910     0.926      4547
weighted avg      0.968     0.968     0.968      4547

Matthews Correlation Coefficient: 0.935
```