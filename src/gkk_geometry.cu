#include "gkk_geometry.cuh"
#include "gkk_material.cuh"

#include <algorithm>

__host__ __device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const {

  vec3 oc = ray.origin() - center;
  vec3 d = ray.direction();
  // computing discriminant for ray-sphere intersection
  float a = dot(d, d);
  float b = 2.0f*dot(d, oc);
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - 4.0f*a*c;
  float t = -1.0f;
  if (discriminant > 0.0f) {
    float x1 = (-b - sqrtf(discriminant))/(2.0f*a);
    float x2 = (-b + sqrtf(discriminant))/(2.0f*a);
    t = fminf(x1, x2);
    if (t > t_min && t < t_max) {
      hrec.t = t;
      hrec.p = ray.point_at_t(t);
      hrec.n = normal_at_p(hrec.p);
      hrec.material_ptr = material_ptr;
      return true;
    }
  }
  return false;
}

__host__ __device__ vec3 Sphere::normal_at_p(const vec3& point) const
{
  return normalize(point - center);
}
