#ifndef GKK_RANDOM_H
#define GKK_RANDOM_H

#include <functional>
#include <random>

#include "gkk_vec.h"

template<typename T>
inline T gkk_random()
{
  static std::uniform_real_distribution<T> distribution(0.0, 1.0);
  static std::mt19937 generator;
  static std::function<T()> rand_generator = std::bind(distribution, generator);
  return rand_generator();
}

vec3 random_in_unit_sphere();
vec3 random_in_unit_disk();

#endif//GKK_RANDOM_H
