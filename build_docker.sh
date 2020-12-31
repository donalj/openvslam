#!/bin/bash
args=("$@")
docker build -t openvslam-server ./viewer/ --build-arg NUM_THREADS=$(nproc)
docker build -t openvslam-desktop -f Dockerfile.desktop . --build-arg NUM_THREADS=$(nproc) $args
