# gamechangerml/scripts/data_transfer

This directory contains scripts to upload/ download data to/ from S3.


## Overview

```
- gamechangerml/scripts/data_transfer
    |--download_eval_data.py        Download evaluation data
```


## Prerequisites:
1. Create a virtual environment with the Python version specified in [setup.py](../../../setup.py). For venv help, see [here](../../../docs/VENV.md).
2. Activate the virtual environment.
3. Install `gamechangerml` (you must do this every time there are updates to this repo).
    a. `cd` to your local `gamechanger-ml` repository.
    b. Run `pip install .`

## Usage
Before running any script:
- Activate the virtual environment (see [prerequisites](#prerequisites)).
- `cd` into your local `gamechanger-ml` repository
- Refresh your token with AWSAML

### download_eval_data.py
```
python gamechangerml/scripts/data_transfer/download_eval.py
```
- You will be prompted to enter information about what dataset to download and where.


