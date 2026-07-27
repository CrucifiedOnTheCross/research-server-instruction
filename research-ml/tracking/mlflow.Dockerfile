FROM python:3.12-slim

RUN groupadd --gid 1006 research \
    && useradd --uid 1000 --gid research --create-home nikita \
    && pip install --no-cache-dir mlflow==3.14.0 pandas PyYAML

WORKDIR /project

ENV GIT_PYTHON_REFRESH=quiet
