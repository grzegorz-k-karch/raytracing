#!/bin/bash

nvcc -gencode arch=compute_70,code=sm_70 -rdc=true main.cu Objects.cu
