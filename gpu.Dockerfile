# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

ARG serving=true

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget vim nano -y

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

RUN mkdir /seizure_prediction
COPY ./requirements.txt /seizure_prediction
COPY ./requirements-serving.txt /seizure_prediction
WORKDIR /seizure_prediction

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
RUN if [ "$serving" = "true" ] ; then pip install -r requirements-serving.txt ; fi
RUN python -m pip cache purge

COPY ./ /seizure_prediction
