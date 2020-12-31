#!/bin/bash
usage() { echo "Usage: $0 [--threads <0-${NUM_THREADS}>] [--viewer <pangolin|socket>]" 1>&2; exit 1; }
warn_threads() { echo "threads cannot be lower than 1 or greater than ${NUM_THREADS}" 1>&2; exit 1; }
warn_viewer() { echo "Invalid option. Options are \"pangolin\" or \"socket\"" 1>&2; exit 1; }
THREADS=${NUM_THREADS}
VIEWER=socket


if ! options=$(getopt -o tv -l threads,viewer -- "$@")
then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

set -- $options

while [ $# -gt 0 ]; 
do
    case $1 in
        -t|--threads)
            THREADS="$2"
            # (( THREADS > ${NUM_THREADS}  || THREADS <0 )) || warn_threads
            ;;
        -v|--viewer)
            VIEWER="$2"
            # (( VIEWER=="pangolin"  || VIEWER=="socket" )) || warn_viewer
            ;;
        (--) shift; break;;
        (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
        (*) break;;
        
    esac
    shift
done

if [ ${VIEWER} == pangolin ]; then
  mkdir -p build && \
    cd build && \
    cmake \
      -DBUILD_WITH_MARCH_NATIVE=OFF \
      -DUSE_PANGOLIN_VIEWER=ON \
      -DUSE_SOCKET_PUBLISHER=OFF \
      -DUSE_STACK_TRACE_LOGGER=ON \
      -DBOW_FRAMEWORK=DBoW2 \
      -DBUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      .. && \
    make -j${THREADS}
else
  mkdir -p build && \
    cd build && \
    cmake \
      -DBUILD_WITH_MARCH_NATIVE=OFF \
      -DUSE_PANGOLIN_VIEWER=OFF \
      -DUSE_SOCKET_PUBLISHER=ON \
      -DUSE_STACK_TRACE_LOGGER=ON \
      -DBOW_FRAMEWORK=DBoW2 \
      -DBUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      .. && \
    make -j${THREADS}
fi
