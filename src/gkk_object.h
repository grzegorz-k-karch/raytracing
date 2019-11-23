#ifndef GKK_OBJECT_H
#define GKK_OBJECT_H

#include "gkk_vec.h"
#include "gkk_ray.h"

class Material;

struct hit_record {
  float t; // time ?
  vec3 p;
  vec3 n;
  Material *material_ptr;
};

class Object {
 public:
  virtual bool hit(const Ray& ray, float t_min, float t_max,
		   hit_record& hrec) const = 0;
};

class ObjectList: public Object {
 public:
  ObjectList() {}
 ObjectList(Object** objects, int num_objects):
  objects(objects), num_objects(num_objects) {}
  
  virtual bool hit(const Ray& ray,
		   float t_min, float t_max,
		   hit_record& hrec) const;
  
  Object **objects;
  int num_objects;
};

#endif//GKK_OBJECT_H
