# cuda10.2) DL frameworkのversionに注意
#   nvidia/cuda:10.2-devel-ubuntu18.04
# FROM nvidia/cuda:11.1-base-ubuntu20.04
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

# install miniconda package
WORKDIR /opt

# download anaconda package and install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2019.10-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH

# make workspace
RUN mkdir /work

# install common python packages
COPY ./requirements.txt /work

# PyTorch
# https://notekunst.hatenablog.com/entry/pytorch-ubuntu-macbookpro
# RUN git clone --recursive http://github.com/pytorch/pytorch
# RUN export USE_CUDA=1 USE_CUDNN=1 TORCH_CUDA_ARCH_LIST="8.6"
# RUN cd pytorch && python setup.py install

RUN pip install --upgrade pip setuptools && \
    pip install -r /work/requirements.txt
    
# https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# set working directory
WORKDIR /work

# jupyter用にportを開放
EXPOSE 8888
EXPOSE 5000
EXPOSE 6006