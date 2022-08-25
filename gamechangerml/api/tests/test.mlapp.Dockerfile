FROM python:3.8-slim
ARG GC_ML_HOST

RUN apt-get update \
    && apt-get install -y \
    libpq-dev python3-dev build-essential
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./gamechangerml/api/tests/requirements.txt /tmp/testrequirements.txt
RUN pip install -r /tmp/testrequirements.txt
COPY ./ /opt/app-root/src
WORKDIR /opt/app-root/src
ENV PYTHONPATH=/opt/app-root/src
RUN pytest ./gamechangerml/api/tests/api_tests.py
ENTRYPOINT ["pytest", "./gamechangerml/src/model_testing"]

