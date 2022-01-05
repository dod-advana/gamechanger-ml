FROM python:3.8

RUN pip install pytest requests numpy

ENTRYPOINT ["pytest", "gamechangerml/api/tests/api_tests.py"]

