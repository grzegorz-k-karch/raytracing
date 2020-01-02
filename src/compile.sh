#!/bin/bash

nvcc -gencode arch=compute_75,code=sm_75 --device-c main.cu gkk_cuda_utils.cu gkk_color.cu gkk_geometry.cu gkk_object.cu gkk_material.cu gkk_random.cu gkk_vec.cu
nvcc main.o gkk_cuda_utils.o gkk_color.o gkk_geometry.o gkk_object.o gkk_material.o gkk_random.o gkk_vec.o -gencode arch=compute_75,code=sm_75 -o a.out
