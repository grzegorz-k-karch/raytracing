#ifndef GKK_RAY_H
#define GKK_RAY_H

class Ray {
 public:
  __device__ Ray() {}
  __device__ Ray(const vec3& o, const vec3& d): o(o), d(d) {}

  __device__ vec3 origin() const { return o; }
  __device__ vec3 direction() const { return d; }
  __device__ vec3 point_at_t(float t) const { return o + t*d; }
  
  vec3 o;
  vec3 d;  
};

#endif//GKK_RAY_H
