#ifndef CAMERA_H
#define CAMERA_H

#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

namespace pt = boost::property_tree;

class Camera {
 public:
  Camera(pt::ptree camera);

  void Init(float3 lookfrom, float3 lookat, float3 up, float fov,
	    float2 aspect, float aperture, float focus_dist,
	    float time0, float time1);

 private:
  float3 m_origin;
  float3 m_lower_left_corner;
  float3 m_horizontal;
  float3 m_vertical;
  float m_lens_radius;
  float m_time0, m_time1;
};

#endif//CAMERA_H
