FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

#Labels as key value pair
LABEL Maintainer="nullquant"

# Any working directory can be chosen as per choice like '/' or '/home' or /usr/src etc
WORKDIR /home

COPY requirements.txt ./

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install --upgrade pip && \
  pip install -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES=all PYTHONPATH="${PYTHONPATH}:${PWD}" CLI_ARGS=""

#CMD instruction should be used to run the software
#contained by your image, along with any arguments.
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
