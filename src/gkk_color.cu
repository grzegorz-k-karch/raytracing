#include "gkk_color.cuh"
#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"

#define GKK_FLOAT_MAX 3.402823e+38

__host__ __device__ vec3 get_plane_color(const Ray& ray)
{
  vec3 unit_direction = normalize(ray.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f - t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
}

// vec3 get_color(const Ray& ray, Object* world, int depth)
// {
//   hit_record hrec;
//   vec3 color;
//   if (world->hit(ray, 0.001f, GKK_FLOAT_MAX, hrec)) {
//     Ray scattered;
//     vec3 attenuation;
//     if (depth < 50 && hrec.material_ptr->scatter(ray, hrec, attenuation, scattered)) {
//       color = attenuation*get_color(scattered, world, depth+1);
//     }
//   }
//   else {
//     color = get_plane_color(ray);
//   }
//   return color;
// }
