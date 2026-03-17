#!/bin/bash

./cmake_clean.sh

cmake \
    -DCXX_FLAGS="-O3" \
    -DLW_INCLUDE_DIR="$(pwd)/../../../Lightweaver/Source/" \
    ..
