#!/bin/bash

docker run -i -t -v $(pwd):/home/user --gpus all nvidia/cuda:10.1-base-ubuntu16.04
