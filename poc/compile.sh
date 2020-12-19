#!/bin/bash

nvcc -gencode arch=compute_70,code=sm_70 main.cu
