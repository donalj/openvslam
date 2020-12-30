#!/bin/bash
DIR=$1
: ${DIR:=00}
./build/run_kitti_slam \
    -v /vocab/orb_vocab/orb_vocab.dbow2 \
    -d /dataset/Kitti/dataset/sequences/$DIR \
    -c example/kitti/KITTI_mono_00-02.yaml