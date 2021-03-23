#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include "GenericMaterial.h"

class Material {
public:
  __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
				  vec3& attenuation, Ray& outRays,
				  curandState* localRandState) const = 0;
};

class Lambertian : public Material {
public:
  __host__ __device__ Lambertian(const vec3& a) : albedo(a) {}
  __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
				  vec3& attenuation, Ray& outRays,
				  curandState* localRandState) const;
  vec3 albedo;
};

class Metal : public Material {
public:
  __device__ Metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}
  __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
				  vec3& attenuation, Ray& outRays,
				  curandState* localRandState) const;
  vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
public:
  __device__ Dielectric(float refIdx) : refIdx(refIdx) {}
  __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
				  vec3& attenuation, Ray& outRays,
				  curandState* localRandState) const;
  float refIdx;
};

__device__ float schlick(float cosine, float refIdx);

#endif//MATERIALS_CUH
