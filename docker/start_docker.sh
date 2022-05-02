#!/bin/bash

docker run --restart always \
-d \
-it \
--name torch-server \
--runtime nvidia \
--ipc=host \
--gpus all \
-v $(pwd):/opt/pytorch \
comojin1994/cu11.2-ubuntu-18.04-pytorch-1.10.0:0.4 \
/bin/bash;

