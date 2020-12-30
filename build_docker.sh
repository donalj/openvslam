#!/bin/bash
args=("$@")
docker build -t openvslam-desktop -f Dockerfile.desktop . --build-arg NUM_THREADS=$(nproc) $args