FROM python:3.6.13-buster
# get gpu enabling dependencies
RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  apt-key add -
RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/ubuntu18.04/nvidia-container-runtime.list | \
  tee /etc/apt/sources.list.d/nvidia-container-runtime.list
RUN apt-get update
RUN apt-get install -y nvidia-container-runtime
CMD nvidia-smi
# set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
# install python deps
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip wheel setuptools
RUN pip3 install awscli
# add source code paths and use volume
RUN mkdir gamechangerml
COPY setup.py gamechangerml/.
COPY requirements.txt gamechangerml/.
COPY dev-requirements.txt gamechangerml/.
COPY README.md gamechangerml/.
# install gamechangerml python module
ENV BASE_APP_VENV_PATH="/opt/gc-venv-current"
RUN python3 -m venv "${BASE_APP_VENV_PATH}" --copies
RUN "${BASE_APP_VENV_PATH}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel
RUN "${BASE_APP_VENV_PATH}/bin/pip" install --no-cache-dir --no-deps -r "gamechangerml/requirements.txt"
# RUN pip3 install gamechangerml/.
COPY . gamechangerml/.
# RUN mkdir gamechanger-ml/gamechangerml
WORKDIR gamechangerml
# ENTRYPOINT sleep 60 
ENTRYPOINT  ["/bin/bash",  "gamechangerml/api/fastapi/startFast.sh", "DEV"]
