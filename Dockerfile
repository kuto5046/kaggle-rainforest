# pytorch versionに注意
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    sudo \
    wget \
    cmake \
    vim \
    git \
    tmux \
    zip \
    unzip \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libsndfile1

# download anaconda package and install anaconda
WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2019.10-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH

# make workspace
RUN mkdir /work

# install common python packages
COPY ./requirements.txt /work

RUN pip install --upgrade pip setuptools && \
    pip install -r /work/requirements.txt
    
# https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# kaggle setup
COPY ./.kaggle/kaggle.json /.kaggle/
RUN chmod 600 /.kaggle/kaggle.json

# set working directory
WORKDIR /work

# jupyter用にportを開放
EXPOSE 8888
EXPOSE 5000
EXPOSE 6006

# add user
ARG UID=1000
RUN useradd -m -u ${UID} docker
USER ${UID}