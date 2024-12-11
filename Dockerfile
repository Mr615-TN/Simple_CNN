from python:3.12-slim-bullseye

ARG SKIP_CRONTAB_SETUP=true
ENV SKIP_CRONTAB_SETUP=$SKIP_CRONTAB_SETUP

ARG DOCKER_BUILD=true
ENV DOCKER_BUILD=$DOCKER_BUILD

run apt-get update

WORKDIR /app

COPY requirements.txt /app/

run pip3 install --no-cache-dir --upgrade pip3 \
    && pip3 install --no-cache-dir -r requirements.txt 

COPY FirstCNN/ /app/

CMD["python3", "firstCNN.py"]
