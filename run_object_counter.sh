#!/usr/bin/env bash

VOLUME_DIR="$HOME/Videos"

docker run -it --rm --gpus all -v $VOLUME_DIR:/Data -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY wooruang/caffe-object-detector:ubuntu18.04-cuda9.2-cudnn7 $@

sudo chown $USER:$USER -R $VOLUME_DIR

