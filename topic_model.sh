LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

source gamechangerml/setup_env.sh DEV
python gamechangerml/scripts/topic_model/topic_model_loadsave.py load

cd gamechangerml/api
docker-compose build
docker-compose up
localhost:5000/docs


gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 900 \
      --timeout 1600 \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug \
      --reload