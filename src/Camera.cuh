#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>
#include <vector_functions.h>

#include "StatusCodes.h"

namespace pt = boost::property_tree;

class Camera {
public:
  // default constructor
  Camera() {};
  // explicit constructor
  Camera(pt::ptree camera);
  // copy constructor
  Camera(const Camera& other) = delete;
  // move constructor
  Camera(Camera&& other);
  // copy assignment operator
  Camera& operator=(const Camera& other) = delete;
  // move assignment operator
  Camera& operator=(const Camera&& other);

  void Init(float3 lookfrom, float3 lookat, float3 up, float fov, float2 aspect,
            float aperture, float focus_dist, float time0, float time1);

  void copyToDevice(Camera *cameraDevice, StatusCodes &status) const;

private:
  float3 m_origin;
  float3 m_lower_left_corner;
  float3 m_horizontal;
  float3 m_vertical;
  float m_lens_radius;
  float m_time0;
  float m_time1;
};

#endif // CAMERA_CUH
