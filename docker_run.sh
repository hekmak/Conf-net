#!/usr/bin/env bash
# Maintained by H.H

nvidia-docker run --name=tensorcvpclak \
    -p 8885:8888 \
    -p 6001:6006 \
    -v [PATH TO KITTI DEPTH COMPLETION DATASET]:/notebooks/dataset \
    -v [PATH TO THE PROJECT FOLDER]:/notebooks/project \
    -it \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    hamidhekmatian/ros:kinetic \
    bash

