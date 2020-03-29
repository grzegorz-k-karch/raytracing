#ifndef GKK_GEOMETRY_CUH
#define GKK_GEOMETRY_CUH

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"
#include "gkk_material.cuh"
#include "gkk_aabb.cuh"

class Sphere: public Object {
public:
  __device__ Sphere(const vec3& center, const float radius, Material* material_ptr) :
    center(center), radius(radius), material_ptr(material_ptr) {}
  
  __device__ bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ bool bbox(float t0, float t1, AABB& output_bbox) const;
  __device__ vec3 normal_at_p(const vec3& point) const;

  vec3 center;
  float radius;
  Material *material_ptr;
};


class MovingSphere: public Object {
public:
  __device__ MovingSphere(const vec3& center0, const vec3& center1,
			  float time0, float time1,
			  const float radius, Material* material_ptr) :
    center0(center0), center1(center1),
    time0(time0), time1(time1),
    radius(radius), material_ptr(material_ptr) {}

  __device__ bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ bool bbox(float t0, float t1, AABB& output_bbox) const;
  __device__ vec3 normal_at_p(const vec3& point, const vec3& center) const;

  __device__ vec3 center_at_time(float timestamp) const;

  vec3 center0, center1;
  float time0, time1;
  float radius;
  Material *material_ptr;
};


class TriangleMesh: public Object {
 public:
  __device__ TriangleMesh(const vec3* point_list, int num_points,
			  const int* triangle_list, int num_triangles,
			  Material* material_ptr, const vec3& bmin, const vec3& bmax) :
    point_list(point_list), num_points(num_points),
    triangle_list(triangle_list), num_triangles(num_triangles),
    material_ptr(material_ptr), _bbox(AABB(bmin, bmax)) {}

  __device__ bool hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const;
  __device__ bool bbox(float t0, float t1, AABB& output_bbox) const;
  __device__ vec3 normal_at_p(const vec3& point,
			      const vec3 vert0,
			      const vec3 vert1,
			      const vec3 vert2) const;
  
  const vec3 *point_list;
  const int *triangle_list;
  int num_points;
  int num_triangles;
  Material *material_ptr;
  const AABB _bbox;
};

#endif//GKK_GEOMETRY_CUH
