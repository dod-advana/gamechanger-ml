#!/usr/bin/env bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -v versionNumber"
   echo -e "\t-v version number to upload to the s3 bucket"
   return 0
  #  exit 0 # Exit script after printing help
}
function download_transformers() {
  # Print helpFunction in case parameters are empty
  if [ -z "$version" ]
  then
    echo "Version number for s3 must be provided to have the script work";
    helpFunction
    return 0
  fi

  local pkg_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" >/dev/null 2>&1 && pwd )"
  local models_dest_dir="$pkg_dir/models"
  local S3_TRANS_MODEL_PATH="s3://advana-data-zone/bronze/gamechanger/models/transformers/v$version/transformers.tar.gz"
  declare -A TransformerDict

  TransformerDict["bert-base-cased-squad2"]="https://huggingface.co/deepset/bert-base-cased-squad2"
  TransformerDict["distilbart-mnli-12-3"]="https://huggingface.co/valhalla/distilbart-mnli-12-3"
  TransformerDict["distilbert-base-uncased-distilled-squad"]="https://huggingface.co/distilbert-base-uncased-distilled-squad"
  TransformerDict["distilroberta-base"]="https://huggingface.co/distilroberta-base"
  TransformerDict["msmarco-distilbert-base-v2"]="https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2"
  
  echo "Creating transformers folder"
  git lfs install
  for repo in "${!TransformerDict[@]}"; do
    if [ ! -d "$models_dest_dir/transformers/$repo" ] ; then
      echo "$repo"
      git clone "${TransformerDict[$repo]}" "$models_dest_dir/transformers/$repo"
    else
      echo "$repo folder already exists"
      rm -r -f "$models_dest_dir/transformers/$repo"
      git clone "${TransformerDict[$repo]}" "$models_dest_dir/transformers/$repo"
    fi
    rm -r -f "$models_dest_dir/transformers/$repo/.git"

    done

  tar -zcvf "$models_dest_dir/transformers.tar.gz" "$models_dest_dir/transformers/"
  echo "uploading to s3 $S3_TRANS_MODEL_PATH"
  aws s3 cp "$models_dest_dir/transformers.tar.gz" $S3_TRANS_MODEL_PATH
}

while getopts v: flag
do
  case "${flag}" in
      v) version=${OPTARG};
  esac
done

download_transformers