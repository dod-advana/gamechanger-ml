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
RUN mkdir gamechanger-ml
COPY setup.py gamechanger-ml/.
COPY requirements.txt gamechanger-ml/.
COPY dev-requirements.txt gamechanger-ml/.
COPY README.md gamechanger-ml/.
# install gamechangerml python module
RUN pip3 install gamechanger-ml/.
RUN mkdir gamechanger-ml/gamechangerml
WORKDIR gamechanger-ml
ENTRYPOINT  ["/bin/bash",  "gamechangerml/api/fastapi/startFast.sh", "DEV"]
