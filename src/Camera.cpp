#include "Camera.h"

#define _USE_MATH_DEFINES
#include <cmath>

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
