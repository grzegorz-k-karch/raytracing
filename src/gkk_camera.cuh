#ifndef GKK_CAMERA_CUH
#define GKK_CAMERA_CUH

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_random.cuh"
#include "gkk_xmlreader.h"

class Camera {
 public:
  __device__ __host__
  Camera(vec3 lookfrom, vec3 lookat, vec3 up, float fov, float aspect,
	 float aperture, float focus_dist, float _time0, float _time1) {
    time0 = _time0;
    time1 = _time1;

    float theta = fov*pi/180.0f;
    float half_height = tanf(theta/2.0f);
    float half_width = aspect*half_height;

    origin = lookfrom;
    lens_radius = aperture/2.0f;

    w = normalize(lookfrom - lookat);
    u = normalize(cross(up, w));
    v = cross(w, u);
    lower_left_corner = origin
      - half_width*focus_dist*u
      - half_height*focus_dist*v
      - focus_dist*w;
    horizontal = 2.0f*half_width*focus_dist*u;
    vertical = 2.0f*half_height*focus_dist*v;
  }

  __host__
  Camera(pt::ptree camera) {
    vec3 lookFrom = string2vec3(camera.get<std::string>("lookFrom.<xmlattr>.value"));
    vec3 lookAt = string2vec3(camera.get<std::string>("lookAt.<xmlattr>.value"));
    vec3 up = string2vec3(camera.get<std::string>("up.<xmlattr>.value"));
    float fov = camera.get<float>("fov.<xmlattr>.value");
    int res_x = camera.get<int>("res_x.<xmlattr>.value");
    int res_y = camera.get<int>("res_y.<xmlattr>.value");    
    float aspect = float(res_x)/float(res_y);
    float aperture = camera.get<float>("fov.<xmlattr>.value");
    float focus_distance = camera.get<float>("fov.<xmlattr>.value");
    if (focus_distance < 0.0f) {
      focus_distance = (lookFrom-lookAt).length();
    }
    float time0 = camera.get<float>("time0.<xmlattr>.value");
    float time1 = camera.get<float>("time1.<xmlattr>.value");
    Camera(lookFrom, lookAt, up, fov, aspect,
	   aperture, focus_distance, time0, time1);
  }

  __device__ Ray get_ray(float s, float t, curandState* local_rand_state) {
    vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
    vec3 offset = u*rd.x() + v*rd.y();
    float timestamp = time0 + curand_uniform(local_rand_state)*(time1 - time0);
    return Ray(origin + offset, lower_left_corner +
	       s*horizontal + t*vertical - origin - offset,
	       timestamp);
  }

  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  float lens_radius;
  float time0, time1;

  const float pi = 3.14159265358979323846;
};

#endif//GKK_CAMERA_CUH
