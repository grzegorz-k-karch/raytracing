#ifndef GKK_COLOR_CUH
#define GKK_COLOR_CUH

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"
#include <curand_kernel.h>

__device__ vec3 get_plane_color(const Ray& ray);

__device__ vec3 get_color(const Ray& ray, Object* world, curandState* local_rand_state);

#endif//GKK_COLOR_CUH

