#ifndef GKK_RANDOM_CUH
#define GKK_RANDOM_CUH

#include "gkk_vec.cuh"

#include <functional>
#include <random>
#include <curand_kernel.h>

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state);
__device__ vec3 random_in_unit_disk(curandState* local_rand_state);

#endif//GKK_RANDOM_CUH
