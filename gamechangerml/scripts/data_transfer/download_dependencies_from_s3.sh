#!/usr/bin/env bash

"""Download the following dependencies from S3:
  - Transformers Folder (env variable: S3_TRANS_MODEL_PATH)
  - Sentence Index (env variable: S3_SENT_INDEX_PATH)
  - QE Model (env variable: S3_QEXP_PATH)
  - JBook QE Model (env variable: S3_QEXP_JBOOK_PATH)
  - Topic Model (env variable: S3_TOPICS_PATH)
  - Data Folder (env variable: S3_ML_DATA_PATH)

Be sure to set up environment variables for s3 by sourcing 
gamechangerml/setup_env.sh if running this manually.
"""

echo "Be sure to set up environment variables for s3 by sourcing gamechangerml/setup_env.sh if running this manually"

function download_and_unpack_deps() {
  
  local pkg_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" >/dev/null 2>&1 && pwd )"  # path to gamechangerml dir
  local models_dest_dir="$pkg_dir/models/"  
  local data_dest_dir="$pkg_dir"

  mkdir -p "$models_dest_dir" "$data_dest_dir"

  # echo "Downloading Transformers Folder"
  echo "S3 MODEL PATH TRANSFORMERS: $S3_TRANS_MODEL_PATH"
  aws s3 cp "$S3_TRANS_MODEL_PATH" "$models_dest_dir" --no-progress

  echo "Downloading Sentence Index"
  echo "S3 MODEL PATH SENTENCE INDEX: $S3_SENT_INDEX_PATH"
  aws s3 cp "$S3_SENT_INDEX_PATH" "$models_dest_dir" --no-progress

  echo "Downloading Title Index"
  echo "S3 MODEL PATH TITLE INDEX: $S3_TITLE_INDEX_PATH"
  aws s3 cp "$S3_TITLE_INDEX_PATH" "$models_dest_dir" --no-progress

  echo "Downloading QE Model"
  echo "S3 QE MODEL: $S3_QEXP_PATH"
  aws s3 cp "$S3_QEXP_PATH" "$models_dest_dir" --no-progress

  echo "Downloading JBOOK QE Model"
  echo "S3 JBOOK QE MODEL: $S3_QEXP_JBOOK_PATH"
  aws s3 cp "$S3_QEXP_JBOOK_PATH" "$models_dest_dir" --no-progress

  echo "Downloading Topic Model"
  echo "S3 TOPIC MODEL: $S3_TOPICS_PATH"
  aws s3 cp "$S3_TOPICS_PATH" "$models_dest_dir" --no-progress

  echo "Downloading Data Folder"
  echo "DATA DIRECTORY: $S3_ML_DATA_PATH"
  aws s3 cp "$S3_ML_DATA_PATH" "$data_dest_dir" --no-progress

  echo "Uncompressing all tar files in models"
  find "$models_dest_dir" -maxdepth 1 -type f -name "*.tar.gz" | while IFS=$'\n' read -r f; do
    tar kxzf "$f" --exclude '*/.git/*' --exclude '*/.DS_Store/*' -C "$models_dest_dir"
  done
  rm "$models_dest_dir"*.tar.gz
  
  # no longer pulling data files right now
  #echo "Uncompressing all tar files in data"
  #find "$data_dest_dir" -maxdepth 1 -type f -name "*.tar.gz" | while IFS=$'\n' read -r f; do
  #  tar xzvf "$f" --exclude '*/.git/*' --exclude '*/.DS_Store/*' -C "$data_dest_dir"
  #done
}

download_and_unpack_deps