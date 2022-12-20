# Scripts for Training and Testing model functionality

---

The scripts contained in this folder can be used to train new Doc2Vec models, as well as test the 
functionality of models.  Please follow the steps below to utilize the scripts.

### manually_train_models.py

**Script Notes:** This script will train a co-occurance based phrase detector 
using Gensim's `Phraser`, as well as a Doc2Vec model using a wrapper around Gensim's Doc2Vec 
implementations.  The models will be trained using `.json` files located in the `Config.CORPUS_DIR` and 
saved in the `Config.MODEL_DIR` which can be defined in the `gamechangerml/src/modelzoo/semantic/D2V_Config.py`.

### using_existing_models.py

**Script Notes:** In the script specifiy the `model_dir` as well as the specific `model_name` and this 
script will load in the phrase detector models, as well as the Doc2Vec model and run a couple of inferences.


### There are other scripts present--for now they can be ignored
