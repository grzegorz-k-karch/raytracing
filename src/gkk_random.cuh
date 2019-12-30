#ifndef GKK_RANDOM_CUH
#define GKK_RANDOM_CUH

#include <functional>
#include <random>

#include "gkk_vec.cuh"

template<typename T>
__host__ __device__ __inline__ T gkk_random()
{
  return 0.5f;
  // static std::uniform_real_distribution<T> distribution(0.0, 1.0);
  // static std::mt19937 generator;
  // static std::function<T()> rand_generator = std::bind(distribution, generator);
  // return rand_generator();
}

__host__ __device__ vec3 random_in_unit_sphere();
__host__ __device__ vec3 random_in_unit_disk();

#endif//GKK_RANDOM_CUH
