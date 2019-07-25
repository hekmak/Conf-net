#!/usr/bin/env bash
# Maintained by H.H

nvidia-docker run --name=tensorcvpclak \
    -p 8887:8888 \
    -p 6005:6006 \
    -v /path/to/dataset:/notebooks/dataset \
    -v /path/to/project:/notebooks/project \
    -it \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    hamidhekmatian/tensorcvpcl:gpu \

