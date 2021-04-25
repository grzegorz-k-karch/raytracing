#define _USE_MATH_DEFINES
#include <cmath>

#include "Camera.cuh"
#include "vector_utils.h"
#include "cuda_utils.cuh"

void Camera::copyToDevice(Camera* cameraDevice, StatusCodes& status) const
{
  status = StatusCodes::NoError;

  status = CCE(cudaMemcpy(cameraDevice, this,
			  sizeof(Camera), cudaMemcpyHostToDevice));
}

__device__
float3 Camera::randomInUnitDisk(curandState* localRandState) const
{
  float3 p;
  do {
    p = 2.0f*make_float3(curand_uniform(localRandState),
			 curand_uniform(localRandState),
			 0.0f) - make_float3(1.0f, 1.0f, 0.0f);
    
  } while (squared_length(p) >= 1.0f);
  return p;  
}
