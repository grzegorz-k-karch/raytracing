#define _USE_MATH_DEFINES
#include <cmath>

#include "Camera.cuh"
#include "nvidia/helper_math.h"
#include "cuda_utils.cuh"
#include "vector_utils.cuh"

void Camera::copyToDevice(Camera* cameraDevice, StatusCodes& status) const
{
  status = StatusCodes::NoError;

  status = CCE(cudaMemcpy(cameraDevice, this,
			  sizeof(Camera), cudaMemcpyHostToDevice));
}

