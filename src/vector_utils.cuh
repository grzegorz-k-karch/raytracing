#ifndef VECTOR_UTILS_CUH
#define VECTOR_UTILS_CUH

#include <cmath>
#include <string>
#include <sstream>
#include <vector_types.h>
#include <vector_functions.h>
#include <curand_kernel.h>
#include "nvidia/helper_math.h"

float3 string2float3(const std::string& s);

float2 string2float2(const std::string& s);

__host__ __device__ inline
float squaredLength(float3 v)
{
  return dot(v, v);
}

__host__ __device__ inline
float3 operator-(float3 v)
{
  return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ inline
bool refract(float3 v, float3 n, float n1OverN2, float3& refracted)
{
  float3 uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - n1OverN2*n1OverN2*(1.0f - dt*dt);
  if (discriminant > 0.0f) {
    refracted = n1OverN2*(uv - n*dt) - n*sqrtf(discriminant);
    return true;
  }
  return false;
}

__device__
float3 randomInUnitSphere(curandState* localRandState);

__device__
float3 randomInUnitDisk(curandState* localRandState);

#endif//VECTOR_UTILS_CUH
