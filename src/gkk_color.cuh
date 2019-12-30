#ifndef GKK_COLOR_CUH
#define GKK_COLOR_CUH

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"

__host__ __device__ vec3 get_plane_color(const Ray& ray);

__host__ __device__ vec3 get_color(const Ray& ray, Object* world, int depth);

#endif//GKK_COLOR_CUH

