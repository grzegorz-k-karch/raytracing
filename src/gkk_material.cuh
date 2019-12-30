#ifndef GKK_MATERIAL_H
#define GKK_MATERIAL_H

#include "gkk_object.cuh"

class Material {
 public:
  __host__ __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
					   vec3& attenuation, Ray& out_rays) const = 0;
};

class Lambertian : public Material {
 public:
  __host__ __device__ Lambertian(const vec3& a) : albedo(a) {}

  __host__ __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;
  
  vec3 albedo;
};

class Metal : public Material {
 public:
  __host__ __device__ Metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}

  __host__ __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;
  
  vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
 public:
  __host__ __device__ Dielectric(float ref_idx) : ref_idx(ref_idx) {}

  __host__ __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;

  float ref_idx;
};

__host__ __device__ float schlick(float cosine, float ref_idx);

#endif//GKK_MATERIAL_H
