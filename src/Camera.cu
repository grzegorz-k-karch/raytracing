#define _USE_MATH_DEFINES
#include <cmath>

#include "Camera.cuh"
#include "nvidia/helper_math.h"
#include "cuda_utils.cuh"
#include "vector_utils.cuh"

void Camera::copyToDevice(Camera* d_camera, StatusCodes& status) const
{
  status = StatusCodes::NoError;

  status = CCE(cudaMemcpy(d_camera, this,
			  sizeof(Camera), cudaMemcpyHostToDevice));
}

