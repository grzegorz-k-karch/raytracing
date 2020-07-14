#ifndef GKK_AABB_H
#define GKK_AABB_H

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }

__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

__device__ inline void swap(float& a, float& b)
{
  float tmp = a;
  a = b;
  b = tmp;
}

class AABB {
public:
  __device__ AABB() {}
  __device__ AABB(const vec3& a, const vec3&b) :
    bmin(a), bmax(b) {}
  __device__ AABB(const AABB& bbox) :
  bmin(bbox.bmin), bmax(bbox.bmax) {}  

  __device__ vec3 min() const { return bmin; }
  __device__ vec3 max() const { return bmax; }

  __device__ bool hit(const Ray& r, float tmin, float tmax) const {
    for (int a = 0; a < 3; a++) {
      float invD = 1.0f/r.direction()[a];
      float t0 = (bmin[a] - r.origin()[a])*invD;
      float t1 = (bmax[a] - r.origin()[a])*invD;
      if (invD < 0.0f) {
	swap(t0, t1);
      }
      tmin = t0 > tmin ? t0 : tmin; 
      tmax = t1 < tmax ? t1 : tmax;
      if (tmax <= tmin) {
	return false;
      }
    }
    return true;
  }

  vec3 bmin;
  vec3 bmax;
};

__device__ inline AABB surrounding_bbox(const AABB& b0, const AABB& b1) {
  vec3 small(ffmin(b0.min().x(), b1.min().x()),
	     ffmin(b0.min().y(), b1.min().y()),
	     ffmin(b0.min().z(), b1.min().z()));
  vec3 big(ffmax(b0.max().x(), b1.max().x()),
	   ffmax(b0.max().y(), b1.max().y()),
	   ffmax(b0.max().z(), b1.max().z()));
  
  return AABB(small, big);
}


#endif//GKK_AABB_H
