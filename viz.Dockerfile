FROM texlive:latest-small
# base image python version >= 3.10.7

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

RUN mkdir /seizure_prediction
COPY ./ /seizure_prediction
WORKDIR /seizure_prediction

## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget vim nano -y

RUN apt install latexmk -y

# install CTAN packages
RUN tlmgr install relsize xecjk

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
RUN python -m pip cache purge
