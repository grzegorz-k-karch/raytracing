#ifndef AABB_CUH
#define AABB_CUH

#include "Ray.cuh"

__device__ __inline__ void swap(float& a, float& b)
{
  float tmp = a;
  a = b;
  b = tmp;
}

class AABB {
public:
  __device__ AABB() {}
  __device__ __host__ AABB(float3 a, float3 b) :
    m_bmin{a.x, a.y, a.z},
    m_bmax{b.x, b.y, b.z} {}
  __device__ AABB(const AABB& bbox) :
    m_bmin{bbox.m_bmin[0], bbox.m_bmin[1], bbox.m_bmin[2]},
    m_bmax{bbox.m_bmax[0], bbox.m_bmax[1], bbox.m_bmax[2]} {}
  __device__ AABB& operator=(const AABB& other) {
    m_bmin[0] = other.min().x;
    m_bmin[1] = other.min().y;
    m_bmin[2] = other.min().z;
    m_bmax[0] = other.max().x;
    m_bmax[1] = other.max().y;
    m_bmax[2] = other.max().z;
    return *this;
  }

  __device__ __host__ float3 min() const {
    return make_float3(m_bmin[0], m_bmin[1], m_bmin[2]);
  }

  __device__ __host__ float3 max() const {
    return make_float3(m_bmax[0], m_bmax[1], m_bmax[2]);
  }

  __device__ bool hit(const Ray& ray, float tMin, float tMax) const;

  friend __device__ __inline__ AABB surroundingBBox(const AABB& b0,
						    const AABB& b1);
private:
  float m_bmin[3];
  float m_bmax[3];
};

__device__ __inline__
AABB surroundingBBox(const AABB& b0, const AABB& b1) {
  float3 small = make_float3(fminf(b0.m_bmin[0], b1.m_bmin[0]),
			     fminf(b0.m_bmin[1], b1.m_bmin[1]),
			     fminf(b0.m_bmin[2], b1.m_bmin[2]));
  float3 big = make_float3(fmaxf(b0.m_bmax[0], b1.m_bmax[0]),
			   fmaxf(b0.m_bmax[1], b1.m_bmax[1]),
			   fmaxf(b0.m_bmax[2], b1.m_bmax[2]));

  return AABB(small, big);
}

#endif//AABB_CUH
