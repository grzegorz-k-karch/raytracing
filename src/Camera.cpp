#include "logging.h"
#include "Camera.cuh"
#include "vector_utils.cuh"

Camera::Camera(pt::ptree camera)
{
  LOG_TRIVIAL(trace) << "explicit constructor";

  float3 up = string2float3(camera.get<std::string>("up.<xmlattr>.value"));
  float fov = camera.get<float>("fov.<xmlattr>.value");
  float aperture = camera.get<float>("aperture.<xmlattr>.value");
  float2 aspect = string2float2(camera.get<std::string>("aspect.<xmlattr>.value"));
  float3 lookFrom = string2float3(camera.get<std::string>("lookFrom.<xmlattr>.value"));
  float3 lookAt = string2float3(camera.get<std::string>("lookAt.<xmlattr>.value"));
  float focus_distance = camera.get<float>("focus_distance.<xmlattr>.value");
  if (focus_distance < 0.0f) {
    focus_distance = length(lookFrom-lookAt);
  }

  Init(lookFrom, lookAt, up, fov, aspect,
       aperture, focus_distance);
}

void Camera::Init(float3 lookfrom, float3 lookat, float3 up, float fov,
		  float2 aspect, float aperture, float focusDist)
{
  m_origin = lookfrom;
  
  m_lensRadius = aperture/2.0f;

  float theta = fov*M_PI/180.0f;
  float halfHeight = tanf(theta/2.0f);
  float halfWidth = halfHeight*aspect.x/aspect.y;
  m_w = normalize(lookfrom - lookat);
  m_u = normalize(cross(up, m_w));
  m_horizontal = 2.0f*halfWidth*focusDist*m_u;

  m_v = cross(m_w, m_u);
  m_vertical = 2.0f*halfHeight*focusDist*m_v;

  m_lowerLeftCorner = m_origin
    - halfWidth*focusDist*m_u
    - halfHeight*focusDist*m_v
    - focusDist*m_w;
}
