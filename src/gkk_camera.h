#ifndef GKK_CAMERA_H
#define GKK_CAMERA_H

#include "gkk_vec.h"
#include "gkk_ray.h"

#include <cmath>

class Camera {
 public:
  Camera(vec3 lookfrom, vec3 lookat, vec3 up, float fov, float aspect) {
    vec3 u, v, w;
    float theta = fov*pi/180.0f;
    float half_height = std::tan(theta/2.0f);
    float half_width = aspect*half_height;
    
    origin = lookfrom;
    w = normalize(lookfrom - lookat);
    u = normalize(cross(up, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width*u - half_height*v - w;
    horizontal = 2.0f*half_width*u;
    vertical = 2.0f*half_height*v;
  }
  Camera(float fov, float aspect) {

    float theta = fov*pi/180.0f;
    float half_height = std::tan(theta/2.0f);
    float half_width = aspect*half_height;
    
    lower_left_corner = vec3(-half_width, -half_height, -1.0f);
    horizontal = vec3(2.0f*half_width, 0.0f, 0.0f);
    vertical = vec3(0.0f, 2.0f*half_height, 0.0f);
    origin = vec3(0.0f, 0.0f, 0.0f);
  }

  Ray get_ray(float s, float t) {
    return Ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
  }
  
  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  const float pi = 3.14159265358979323846;
};

#endif//GKK_CAMERA_H
