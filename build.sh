#!/bin/bash

echo "Start setup"
if [[ -d "./build/" ]]; then
    rm -rf ./build/
fi
mkdir ./build/
echo "Setup finished"
echo "Build programme in build"
eval "cmake -B build"
echo "Compile programme in build"
cd build
mkdir result
eval "make" 2> "../build.log"
cd ..
echo "Build finish"
echo "See build.log"