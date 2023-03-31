FROM debian:stretch

RUN    apt-get update  -y  && apt-get install -y software-properties-common 
RUN    apt-get clean
RUN    apt-get install ffmpeg libsm6 libxext6  -y
RUN    apt-get install -y wget
RUN    apt-get install build-essential -y                  
RUN    apt-get install -y git 

ENV    CONDA_DIR /opt/conda
RUN    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV    PATH=$CONDA_DIR/bin:$PATH
## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN pip install joblib
RUN pip install tensorflow
RUN pip install DateTime
## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

# To run in Duke Server
RUN mkdir /app /data /config
RUN addgroup -gid 4000 railabs
RUN test "$(getent passwd ea184)" || useradd --create-home --gid 4000  --uid 1513544 --shell /bin/bash ea184    
RUN chown  ea184:railabs /app /data /config
USER ea184