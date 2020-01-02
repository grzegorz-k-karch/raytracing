#include "gkk_color.cuh"
#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"
#include "gkk_material.cuh"

#define GKK_FLOAT_MAX 3.402823e+38

__host__ __device__ vec3 get_plane_color(const Ray& ray)
{
  vec3 unit_direction = normalize(ray.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f - t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
}

__host__ __device__ vec3 get_color(const Ray& ray, Object* world, int depth)
{
  hit_record hrec;
  vec3 color;
  if (world->hit(ray, 0.001f, GKK_FLOAT_MAX, hrec)) {
    Ray scattered;
    vec3 attenuation;
    if (depth < 5 && hrec.material_ptr->scatter(ray, hrec, attenuation, scattered)) { // TODO: 50
      color = attenuation*get_color(scattered, world, depth+1);
    }
  }
  else {
    color = get_plane_color(ray);
  }
  return color;
}

__host__ __device__ vec3 get_color(const Ray& ray, Object* world)
{
  hit_record hrec;
  vec3 color;
  Ray in_ray = ray;
  vec3 attenuation_total = vec3(1.0f, 1.0f, 1.0f);

  for (int i = 0; i < 50; i++) {
    if (world->hit(ray, 0.001f, GKK_FLOAT_MAX, hrec)) {
      vec3 attenuation;
      Ray scattered;
      if (hrec.material_ptr->scatter(in_ray, hrec, attenuation, scattered)) {
    	attenuation_total *= attenuation;
    	in_ray = scattered;
      }
    }
    else {
      color = get_plane_color(in_ray);
      break;
    }
  }

  color *= attenuation_total;

  return color;
}
