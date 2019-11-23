#ifndef GKK_GEOMETRY_H
#define GKK_GEOMETRY_H

#include "gkk_vec.h"
#include "gkk_ray.h"
#include "gkk_object.h"
#include "gkk_material.h"

class Sphere: public Object {
 public:
 Sphere(const vec3& center, const float radius, Material* material_ptr) :
  center(center), radius(radius), material_ptr(material_ptr) {}
  
  bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  vec3 normal_at_p(const vec3& point) const;

  vec3 center;
  float radius;
  Material *material_ptr;
};

#endif//GKK_GEOMETRY_H
