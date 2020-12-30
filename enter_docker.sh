#!/bin/bash
NAME=$1
: ${NAME:=donal}

if [ ! "$(docker ps -q -f name=${NAME})" ]; then
    echo "No running container found."
    if [ "$(docker ps -aq -f status=exited -f name=${NAME})" ]; then
        echo "Stopped container exists. Removing"
        docker rm ${NAME}
    fi
    echo "Creating container with name: ${NAME}"
    docker run --runtime=nvidia -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
        --volume /data/Datasets/:/dataset:ro \
        --volume /data/Vocab:/vocab:ro \
        --volume $(pwd):/openvslam \
        --name ${NAME} \
        openvslam-desktop
fi
echo "Container found. Entering"
docker exec -it ${NAME} bash

