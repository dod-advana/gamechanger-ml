FROM python:3.8-slim

RUN pip install pytest requests numpy
COPY ./ /opt/app-root/src
WORKDIR /opt/app-root/src
ENV PYTHONPATH=/opt/app-root/src
ARG GC_ML_HOST="http://host.docker.internal"
RUN pytest ./gamechangerml/api/tests/api_tests.py
# ENTRYPOINT ["pytest", "./gamechangerml/api/tests/api_tests.py"]

