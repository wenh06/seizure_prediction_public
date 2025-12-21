FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

ARG serving=true
ARG torch=false
ARG port=11111
ARG AptSource=tsinghua

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"


# if AptSource is tsinghua, then copy misc/debian.souces
# to cover /etc/apt/sources.list.d/debian.sources
# NOTE that the source is uses debian 12 (bookworm) source
COPY misc/debian.sources /tmp/debian.sources
RUN if [ "$AptSource" = "tsinghua" ] ; \
    then cp /tmp/debian.sources /etc/apt/sources.list.d/debian.sources ; \
    fi


## Install your dependencies here using apt install, etc.

RUN apt update
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget vim -y

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

RUN if [ "$torch" = "true" ] ; then pip install torch==2.0.1 ; fi

RUN mkdir /seizure_prediction
COPY ./requirements.txt /seizure_prediction
COPY ./requirements-serving.txt /seizure_prediction
WORKDIR /seizure_prediction

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN if [ "$AptSource" = "tsinghua" ] ; \
    then pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple ; \
    fi
RUN pip install -r requirements.txt
RUN if [ "$serving" = "true" ] ; then pip install -r requirements-serving.txt ; fi
RUN python -m pip cache purge

COPY ./ /seizure_prediction

# if serving, then expose port 11111, and run the serving script
EXPOSE $port


# command to build the image:
# docker build -t seizure-prediction-service:latest -f base.Dockerfile .

# command to run the image (with serving):
# docker run -d -p 11111:11111 seizure-prediction-service:latest bash -c "python service.py --ip 0.0.0.0 --port 11111"
