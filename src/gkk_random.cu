#include "gkk_random.cuh"


__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
  vec3 p;
  do {
    p = 2.0f*vec3(curand_uniform(local_rand_state),
		  curand_uniform(local_rand_state),
		  curand_uniform(local_rand_state)) - vec3(1.0f, 1.0f, 1.0f);
    
  } while (p.squared_length() >= 1.0f);
  return p;
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state)
{
  vec3 p;
  do {
    p = 2.0f*vec3(curand_uniform(local_rand_state),
		  curand_uniform(local_rand_state),
		  0.0f) - vec3(1.0f, 1.0f, 0.0f);
    
  } while (p.squared_length() >= 1.0f);
  return p;
}
