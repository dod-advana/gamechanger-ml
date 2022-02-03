#!/bin/bash
echo "Be sure to set up environment variables for s3 by sourcing setup_env.sh if running this manually"
echo "Downloading Transformers Folder"
echo "S3 MODEL PATH TRANSFORMERS: $S3_TRANS_MODEL_PATH"
#python -c "from gamechangerml.src.utilities.utils import get_transformer_cache; get_transformer_cache(model_path='$S3_TRANS_MODEL_PATH', overwrite=False)"
aws s3 cp "$S3_TRANS_MODEL_PATH" $PWD/gamechangerml/models/.

echo "Downloading Data Folder"
echo "DATA DIRECTORY: $S3_ML_DATA_PATH"
#python -c "from gamechangerml.src.utilities.utils import get_transformer_cache; get_transformer_cache(model_path='$S3_TRANS_MODEL_PATH', overwrite=False)"
aws s3 cp "$S3_ML_DATA_PATH" $PWD/gamechangerml/data/.

echo "Downloading Sentence Index"
echo "S3 MODEL PATH SENTENCE INDEX: $S3_SENT_INDEX_PATH"
aws s3 cp "$S3_SENT_INDEX_PATH" $PWD/gamechangerml/models/.
#python -c "from gamechangerml.src.utilities.utils import get_sentence_index; get_sentence_index(model_path='$S3_SENT_INDEX_PATH',overwrite=False)"

echo "Downloading QE Model"
echo "S3 QE MODEL: $S3_QEXP_PATH"
aws s3 cp "$S3_QEXP_PATH" $PWD/gamechangerml/models/.

echo "Downloading JBOOK QE Model"
echo "S3 JBOOK QE MODEL: $S3_QEXP_JBOOK_PATH"
aws s3 cp "$S3_QEXP_JBOOK_PATH" $PWD/gamechangerml/models/.

echo "Downloading NGRAM QE Model"
echo "S3 NGRAM QE MODEL: $S3_QEXP_NGRAM_PATH"
aws s3 cp "$S3_QEXP_NGRAM_PATH" $PWD/gamechangerml/models/.

echo "Downloading Topic Model"
echo "S3 TOPIC MODEL: $S3_TOPICS_PATH"
aws s3 cp "$S3_TOPICS_PATH" $PWD/gamechangerml/models/topic_models/

echo "Uncompressing all tar files in models"
for f in ./gamechangerml/models/*.tar.gz; do
  tar kxvfz "$f" --exclude '*/.git/*' --exclude '*/.DS_Store/*' -C ./gamechangerml/models/;
done 

echo "Uncompressing all tar files in data"
for f in ./gamechangerml/data/*.tar.gz; do
  tar kxvfz "$f" --exclude '*/.git/*' --exclude '*/.DS_Store/*' -C ./gamechangerml/data/;
done