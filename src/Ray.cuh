#ifndef RAY_CUH
#define RAY_CUH

#include <curand_kernel.h>
#include "nvidia/helper_math.h"

class Material;

struct HitRecord {
  float t; // time
  float3 p; // position
  float3 n; // normal at p
  const Material *material;
};

class Ray {
public:
  __device__ Ray() {}
  __device__ Ray(float3 origin, float3 direction, float timestamp = 0.0f)
    : m_origin(origin), m_direction(direction), m_timestamp(timestamp) {}

  __device__ float3 pointAtT(float t) const {
    return m_origin + t * m_direction;
  }

  __device__ __inline__ float direction(int d) const {
    if (d == 0) {
      return m_direction.x;
    }
    else if (d == 1) {
      return m_direction.y;
    }
    else {
      return m_direction.z;
    }
  }

  __device__ __inline__ float origin(int d) const {
    if (d == 0) {
      return m_origin.x;
    }
    else if (d == 1) {
      return m_origin.y;
    }
    else {
      return m_origin.z;
    }
  }

  float3 m_origin;
  float3 m_direction;
  float m_timestamp;
};

#endif//RAY_CUH
