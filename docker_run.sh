#!/usr/bin/env bash
# Maintained by H.H

nvidia-docker run --name=tensorcvpclak \
    -p 8885:8888 \
    -p 6001:6006 \
    -v /media/hamidhekmatian/My4TBHD1/Datasets/Kitti_depth:/notebooks/dataset \
    -v /media/hamidhekmatian/My4TBHD1/AI_git/AI/depth_completion:/notebooks/project \
    -it \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    hamidhekmatian/ros:kinetic \
    bash

