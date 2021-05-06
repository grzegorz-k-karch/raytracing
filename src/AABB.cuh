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
    bmin{a.x, a.y, a.z},
    bmax{b.x, b.y, b.z} {}
  __device__ AABB(const AABB& bbox) :
    bmin{bbox.bmin[0], bbox.bmin[1], bbox.bmin[2]},
    bmax{bbox.bmax[0], bbox.bmax[1], bbox.bmax[2]} {}

  __device__ __host__ float3 min() const { return make_float3(bmin[0], bmin[1], bmin[2]); }
  __device__ __host__ float3 max() const { return make_float3(bmax[0], bmax[1], bmax[2]); }

  __device__ bool hit(const Ray& ray, float tMin, float tMax) const;

  friend __device__ __inline__
  AABB surroundingBBox(const AABB& b0, const AABB& b1);
  
private:
  float bmin[3];
  float bmax[3];
};

// __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }

// __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

__device__ __inline__
AABB surroundingBBox(const AABB& b0, const AABB& b1) {
  float3 small = make_float3(fminf(b0.bmin[0], b1.bmin[0]),
			     fminf(b0.bmin[1], b1.bmin[1]),
			     fminf(b0.bmin[2], b1.bmin[2]));
  float3 big = make_float3(fmaxf(b0.bmax[0], b1.bmax[0]),
			   fmaxf(b0.bmax[1], b1.bmax[1]),
			   fmaxf(b0.bmax[2], b1.bmax[2]));
  
  return AABB(small, big);
}



#endif//AABB_CUH
