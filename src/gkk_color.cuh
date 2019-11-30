#ifndef GKK_COLOR_H
#define GKK_COLOR_H

#include "gkk_vec.cuh"
#include "gkk_ray.cuh"
#include "gkk_object.cuh"

vec3 get_plane_color(const Ray& ray);

vec3 get_color(const Ray& ray, Object* world, int depth);

#endif//GKK_COLOR_H

