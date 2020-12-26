FROM ubuntu:18.04 
 
# update
RUN apt-get -y update && apt-get install -y \
sudo \
wget \
vim \
git \
tmux \
zip \
unzip \
libsndfile1

WORKDIR /opt

# download anaconda package and install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2019.10-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH

# make for working directry
RUN mkdir /work

# update pip
COPY ./requirements.txt /work

RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

RUN pip install --upgrade pip && \
    pip install -r /work/requirements.txt

WORKDIR /work
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]