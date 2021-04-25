#ifndef RAY_CUH
#define RAY_CUH

#include <curand_kernel.h>
#include "vector_utils.h"

class Ray {
 public:
  __device__ Ray() {}
  __device__ Ray(float3 origin,
		 float3 direction,
		 float timestamp=0.0f) :
    m_origin(origin),
    m_direction(direction),
    m_timestamp(timestamp) {}

  __device__ float3 pointAtT(float t) const {
    return m_origin + t*m_direction;
  }

  float3 m_origin;
  float3 m_direction;
  float m_timestamp;
};

#endif//RAY_CUH
