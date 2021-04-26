#include "vector_utils.cuh"


__device__
float3 randomInUnitSphere(curandState* localRandState)
{
  float3 p;
  do {
    p = 2.0f*make_float3(curand_uniform(localRandState),
			 curand_uniform(localRandState),
			 curand_uniform(localRandState)) - make_float3(1.0f, 1.0f, 1.0f);
    
  } while (squaredLength(p) >= 1.0f);
  return p;
}


__device__
float3 randomInUnitDisk(curandState* localRandState)
{
  float3 p;
  do {
    p = 2.0f*make_float3(curand_uniform(localRandState),
			 curand_uniform(localRandState),
			 0.0f) - make_float3(1.0f, 1.0f, 0.0f);
    
  } while (squaredLength(p) >= 1.0f);
  return p;  
}

