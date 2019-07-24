FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER llluckygirl <rhyszh@163.com>
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
# install basic dependencies
RUN apt-get update 
RUN apt-get install -y wget \
		vim \
		cmake

# install Anaconda3
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-

Linux-x86_64.sh -O ~/anaconda3.sh
RUN bash ~/anaconda3.sh -b -p /home/anaconda3 \
	&& rm ~/anaconda3.sh 
ENV PATH /home/anaconda3/bin:$PATH

# change mirror
RUN mkdir ~/.pip \
	&& cd ~/.pip 	
RUN	echo -e "[global]\nindex-url = https://pypi.mirrors.ustc.edu.cn/simple/" 

>> ~/pip.conf

# install tensorflow
RUN /home/anaconda3/bin/pip install lightgbm

