#!/bin/sh

conda create -n rapids-22.12 -c rapidsai -c conda-forge -c nvidia cudf=22.12 python=3.8 cudatoolkit=11.5
