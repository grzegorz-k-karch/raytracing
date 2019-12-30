#include "gkk_random.cuh"

__host__ __device__ vec3 random_in_unit_sphere()
{
  vec3 p;
  do {
    p = 2.0f*vec3(gkk_random<float>(),
		  gkk_random<float>(),
		  gkk_random<float>()) - vec3(1.0f, 1.0f, 1.0f);
    
  } while (p.squared_length() >= 1.0f);
  return p;
}

__host__ __device__ vec3 random_in_unit_disk()
{
  vec3 p;
  do {
    p = 2.0f*vec3(gkk_random<float>(),
		  gkk_random<float>(),
		  0.0f) - vec3(1.0f, 1.0f, 0.0f);
    
  } while (p.squared_length() >= 1.0f);
  return p;
}
