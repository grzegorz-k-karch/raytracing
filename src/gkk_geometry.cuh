#ifndef GKK_GEOMETRY_CUH
#define GKK_GEOMETRY_CUH

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"
#include "gkk_material.cuh"

class Sphere: public Object {
 public:
  __device__ Sphere(const vec3& center, const float radius, Material* material_ptr) :
    center(center), radius(radius), material_ptr(material_ptr) {}
  
  __device__ bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ vec3 normal_at_p(const vec3& point) const;

  vec3 center;
  float radius;
  Material *material_ptr;
};

#endif//GKK_GEOMETRY_CUH
