#ifndef GKK_MATERIAL_H
#define GKK_MATERIAL_H

#include "gkk_object.h"

class Material {
 public:
  virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const = 0;
};

class Lambertian : public Material {
 public:
 Lambertian(const vec3& a) : albedo(a) {}

  virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;
  
  vec3 albedo;
};

class Metal : public Material {
 public:
 Metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}

  virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;
  
  vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
 public:
 Dielectric(float ref_idx) : ref_idx(ref_idx) {}

  virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
		       vec3& attenuation, Ray& out_rays) const;

  float ref_idx;
};

float schlick(float cosine, float ref_idx);

#endif//GKK_MATERIAL_H
