#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include "vector_utils.cuh"

#include "Ray.cuh"
#include "StatusCode.h"

namespace pt = boost::property_tree;

class Camera {
public:

  Camera() = default;  // default constructor
  // Camera(const Camera &other) {}  // copy constructor
  Camera(pt::ptree camera);  // explicit constructor

  void Init(float3 lookfrom, float3 lookat, float3 up,
	    float fov, float2 aspect,
            float aperture, float focusDist);

  void copyToDevice(Camera *d_camera, StatusCode &status) const;

  __device__
  Ray getRay(float s, float t, curandState* localRandState) const {
    float3 randomInLensDisk = m_lensRadius*randomInUnitDisk(localRandState);
    float3 offset = m_u*randomInLensDisk.x + m_v*randomInLensDisk.y;
    return Ray(m_origin + offset, m_lowerLeftCorner +
	       s*m_horizontal + t*m_vertical - m_origin - offset);
  }
  __device__
  Ray getRay(float s, float t) const {
    return Ray(m_origin, m_lowerLeftCorner +
	       s*m_horizontal + t*m_vertical - m_origin);
  }

public: // FIXME
  float3 m_origin;
  float3 m_lowerLeftCorner;
  float3 m_horizontal;
  float3 m_vertical;
  float3 m_u, m_v, m_w;
  float m_lensRadius;
};

#endif // CAMERA_CUH
