#ifndef GKK_MATERIAL_H
#define GKK_MATERIAL_H

#include "gkk_object.cuh"

#include <curand_kernel.h>

class Material {
 public:
  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const = 0;
};

class Lambertian : public Material {
 public:
  __device__ Lambertian(const vec3& a) : albedo(a) {}

  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;
  
  vec3 albedo;
};

class Metal : public Material {
 public:
  __device__ Metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}

  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;
  
  vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
 public:
  __device__ Dielectric(float ref_idx) : ref_idx(ref_idx) {}

  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;

  float ref_idx;
};

__device__ float schlick(float cosine, float ref_idx);

#endif//GKK_MATERIAL_H
