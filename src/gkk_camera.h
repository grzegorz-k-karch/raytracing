#ifndef GKK_CAMERA_H
#define GKK_CAMERA_H

#include "gkk_vec.h"
#include "gkk_ray.h"

#include <cmath>

class Camera {
 public:
  Camera(float fov, float aspect) {

    float theta = fov*pi/180.0f;
    float half_height = std::tan(theta/2.0f);
    float half_width = aspect*half_height;
    
    lower_left_corner = vec3(-half_width, -half_height, -1.0f);
    horizontal = vec3(2.0f*half_width, 0.0f, 0.0f);
    vertical = vec3(0.0f, 2.0f*half_height, 0.0f);
    origin = vec3(0.0f, 0.0f, 0.0f);
  }

  Ray get_ray(float u, float v) {
    return Ray(origin, lower_left_corner + u*horizontal + v*vertical);
  }
  
  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  const float pi = 3.14159265358979323846;
};

#endif//GKK_CAMERA_H
