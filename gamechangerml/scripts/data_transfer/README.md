# gamechangerml/scripts/data_transfer

This directory contains scripts to upload/ download data to/ from S3.

## Directory Overview

```
├── gamechangerml/scripts/data_transfer
│   ├── download_eval_data.py                Download evaluation data
│   ├── download_dependencies_from_s3.sh     Download model/ data dependencies from S3
│   ├── read_write_transformers_cache_s3.sh  Read or write the transformers cache in S3
│   ├── upload_transformers_repo_to_s3.sh    Clone transformers repos and upload them to S3
│   ├── topic_model
│   │   ├── topic_model_loadsave_s3.sh       Load/ save topic model in S3 (using Python)
│   │   └── topic_model_loadsave_s3.py       Load/ save topic model in S3 (using Bash)
│   ├── corpus
│   │   ├── download_corpus_s3.py            Download the corpus (using Python)
│   │   └── download_corpus_s3.sh            Download the corpus (using Bash)
```

## Prerequisites for Python Scripts:

1. Create a virtual environment with the Python version specified in [setup.py](../../../setup.py). For venv help, see [here](../../../docs/VENV.md).
2. Activate the virtual environment.
3. Install `gamechangerml` (you must do this every time there are updates to this repo).
   a. `cd` to your local `gamechanger-ml` repository.
   b. Run `pip install .`

## Python Script Usage

Before running a Python script:

- Activate the virtual environment (see [prerequisites](#prerequisites)).
- `cd` into your local `gamechanger-ml` repository
- Refresh your token with AWSAML

## Notes

- `download_eval_data.py`: You will be prompted to enter information about what dataset to download and where.

- Another resource for data transfer operations is [gamechangerml/src/data_transfer](../../../gamechangerml/src/data_transfer/).
