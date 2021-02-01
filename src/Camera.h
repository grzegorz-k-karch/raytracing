#ifndef CAMERA_H
#define CAMERA_H

#include <boost/property_tree/xml_parser.hpp>
#include "vector_utils.h"

namespace pt = boost::property_tree;

class Camera {
 public:
  Camera(pt::ptree camera) {
    
    float3 lookFrom =
      string2float3(camera.get<std::string>("lookFrom.<xmlattr>.value"));
    float3 lookAt =
      string2float3(camera.get<std::string>("lookAt.<xmlattr>.value"));
    float3 up =
      string2float3(camera.get<std::string>("up.<xmlattr>.value"));
    float fov =
      camera.get<float>("fov.<xmlattr>.value");
    float aperture =
      camera.get<float>("aperture.<xmlattr>.value");
    float focus_distance =
      camera.get<float>("focus_distance.<xmlattr>.value");
    if (focus_distance < 0.0f) {
      focus_distance = length(lookFrom-lookAt);
    }
    float time0 = camera.get<float>("time0.<xmlattr>.value");
    float time1 = camera.get<float>("time1.<xmlattr>.value");
    
    float2 aspect =
      string2float2(camera.get<std::string>("aspect.<xmlattr>.value"));
    
    Init(lookFrom, lookAt, up, fov, aspect,
	 aperture, focus_distance, time0, time1);
  }

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
