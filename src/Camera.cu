#define _USE_MATH_DEFINES
#include <cmath>

#include "Camera.cuh"
#include "vector_utils.h"


Camera::Camera(pt::ptree camera)
{
  float3 up = string2float3(camera.get<std::string>("up.<xmlattr>.value"));
  float fov = camera.get<float>("fov.<xmlattr>.value");
  float aperture = camera.get<float>("aperture.<xmlattr>.value");
  float time0 = camera.get<float>("time0.<xmlattr>.value");
  float time1 = camera.get<float>("time1.<xmlattr>.value");
  float2 aspect = string2float2(camera.get<std::string>("aspect.<xmlattr>.value"));
  float3 lookFrom = string2float3(camera.get<std::string>("lookFrom.<xmlattr>.value"));
  float3 lookAt = string2float3(camera.get<std::string>("lookAt.<xmlattr>.value"));
  float focus_distance = camera.get<float>("focus_distance.<xmlattr>.value");
  if (focus_distance < 0.0f) {
    focus_distance = length(lookFrom-lookAt);
  }    
    
  Init(lookFrom, lookAt, up, fov, aspect,
       aperture, focus_distance, time0, time1);
}

void Camera::Init(float3 lookfrom, float3 lookat, float3 up, float fov,
		  float2 aspect, float aperture, float focus_dist,
		  float time0, float time1)
{
  m_time0 = time0;
  m_time1 = time1;
  m_origin = lookfrom;
  m_lens_radius = aperture/2.0f;

  float theta = fov*M_PI/180.0f;
  float half_height = tanf(theta/2.0f);
  float half_width = half_height*aspect.x/aspect.y;
  float3 w = normalize(lookfrom - lookat);
  float3 u = normalize(cross(up, w));
  m_horizontal = 2.0f*half_width*focus_dist*u;

  float3 v = cross(w, u);  
  m_vertical = 2.0f*half_height*focus_dist*v;  

  m_lower_left_corner = m_origin
    - half_width*focus_dist*u
    - half_height*focus_dist*v
    - focus_dist*w;
}
