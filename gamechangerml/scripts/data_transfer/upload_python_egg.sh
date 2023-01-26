s3_path="s3://advana-data-zone/bronze/ml/eggs/$GIT_BRANCH"
python setup.py bdist_egg

echo "[INFO] Uploading Python Egg to ${s3_path} ..."
echo $GIT_BRANCH

aws s3 cp dist/gamechangerml-1.10.0-py3.8.egg ${s3_path}

echo "[INFO] Uploaded Python Egg!"