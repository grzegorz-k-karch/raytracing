#ifndef GKK_OBJECT_H
#define GKK_OBJECT_H

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_aabb.cuh"

#include <curand_kernel.h>

class Material;


struct hit_record {
  float t;
  vec3 p;
  vec3 n;
  Material *material_ptr;
};


class Object {
 public:
  __device__ virtual bool hit(const Ray& ray, float t_min, float t_max,
				       hit_record& hrec) const = 0;
  __device__ virtual bool get_bbox(float t0, float t1, AABB& output_bbox) const = 0;
};


class ObjectList: public Object {
 public:
  __device__ ObjectList() :
    objects(nullptr), num_objects(0) {}
  __device__ ObjectList(Object** objects, int num_objects) :
    objects(objects), num_objects(num_objects) {}

  __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ virtual bool get_bbox(float t0, float t1, AABB& output_bbox) const;

  Object **objects;
  int num_objects;

  AABB *bbox;
};


class BVHNode : public Object {
public:
  __device__ BVHNode() {}
  __device__ BVHNode(ObjectList& object_list, float time0, float time1, curandState* rand_state) :
    BVHNode(object_list.objects, 0, object_list.num_objects, time0, time1, rand_state) {}
  __device__ BVHNode(Object** objects, int start, int end, float time0, float time1,
  		     curandState* rand_state);

  __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ virtual bool get_bbox(float t0, float t1, AABB& output_bbox) const;

public:
  Object *left;
  Object *right;

  AABB *bbox;
};

__device__ inline bool compare_bboxes(Object* a, Object* b, int axis)
{
  AABB bbox_a;
  AABB bbox_b;

  if (!a->get_bbox(0.0f, 0.0f, bbox_a) || !b->get_bbox(0.0f, 0.0f, bbox_b)) {
    return false;
  }
  return bbox_a.min().e[axis] <  bbox_b.min().e[axis]; 
}


#endif//GKK_OBJECT_H
