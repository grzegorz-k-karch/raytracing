#ifndef GKK_GEOMETRY_CUH
#define GKK_GEOMETRY_CUH

#include "gkk_xmlreader.h"
#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"
#include "gkk_material.cuh"
#include "gkk_aabb.cuh"

class Sphere: public Object {
public:
  __device__ Sphere(const vec3& center, const float radius,
		    Material* material, MaterialType material_type) :
    center(center), radius(radius), material(material),
    material_type(material_type) {}
  __host__ Sphere(pt::ptree mesh);
  __host__ void copy_to_device(Object** d_obj_list, int list_offset) const;
  __device__ bool hit(const Ray& ray, float t_min,
		      float t_max, hit_record& hrec) const;
  __device__ bool get_bbox(float t0, float t1, AABB& output_bbox) const;
  __device__ vec3 normal_at_p(const vec3& point) const;

  vec3 center;
  float radius;
  Material *material;
  MaterialType material_type;
};


class MovingSphere: public Object {
public:
  __device__ MovingSphere(const vec3& center0, const vec3& center1,
			  float time0, float time1,
			  const float radius, Material* material) :
    center0(center0), center1(center1),
    time0(time0), time1(time1),
    radius(radius), material(material) {}

  __device__ bool hit(const Ray& ray, float t_min,
		      float t_max, hit_record& hrec) const;

  __device__ bool get_bbox(float t0, float t1, AABB& output_bbox) const;

  __device__ vec3 normal_at_p(const vec3& point, const vec3& center) const;

  __device__ vec3 center_at_time(float timestamp) const;

  vec3 center0, center1;
  float time0, time1;
  float radius;
  Material *material;
};


class TriangleMesh: public Object {
 public:
  __device__ TriangleMesh(vec3* point_list, int num_points,
			  int* triangle_list, int num_triangles,
			  Material* material, AABB bbox) :
    point_list(point_list), num_points(num_points),
    triangle_list(triangle_list), num_triangles(num_triangles),
    material(material), bbox(bbox) {}

  __host__ TriangleMesh(pt::ptree mesh);

  __host__ void copy_to_device(Object** d_obj_list, int list_offset) const;

  __device__ bool hit(const Ray& ray, float t_min, float t_max,
		      hit_record& hrec) const;

  __device__ bool get_bbox(float t0, float t1, AABB& output_bbox) const;

  __device__ vec3 normal_at_p(const vec3& point, const vec3 vert0,
			      const vec3 vert1, const vec3 vert2) const;

  vec3 *point_list;
  int *triangle_list;
  int num_points;
  int num_triangles;
  Material *material;
  AABB bbox;
};

#endif//GKK_GEOMETRY_CUH
