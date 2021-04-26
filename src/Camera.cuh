#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include "vector_utils.cuh"

#include "Ray.cuh"
#include "StatusCodes.h"

namespace pt = boost::property_tree;

class Camera {
public:
  // default constructor
  Camera() = default;
  // copy constructor
  Camera(const Camera &other) {
#if __CUDA_ARCH__ >= 200
    printf("camera copy constructor\n");
#endif
  }
  // explicit constructor
  Camera(pt::ptree camera);

  void Init(float3 lookfrom, float3 lookat, float3 up,
	    float fov, float2 aspect,
            float aperture, float focusDist,
	    float time0, float time1);

  void copyToDevice(Camera *cameraDevice, StatusCodes &status) const;

  __device__
  Ray getRay(float s, float t, curandState* localRandState) const {
    float3 randomInLensDisk = m_lensRadius*randomInUnitDisk(localRandState);
    float3 offset = m_u*randomInLensDisk.x + m_v*randomInLensDisk.y;
    float timestamp = m_time0 + curand_uniform(localRandState)*(m_time1 - m_time0);
    return Ray(m_origin + offset, m_lowerLeftCorner +
	       s*m_horizontal + t*m_vertical - m_origin - offset,
	       timestamp);
  }

private:

  float3 m_origin;
  float3 m_lowerLeftCorner;
  float3 m_horizontal;
  float3 m_vertical;
  float3 m_u, m_v, m_w;
  float m_lensRadius;
  float m_time0;
  float m_time1;
};

#endif // CAMERA_CUH
