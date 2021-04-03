#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include "GenericMaterial.h"

class Material {
public:
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const = 0;
};

class Lambertian : public Material {
public:
  __device__ Lambertian() {}
  __device__ Lambertian(const float3& a) : albedo(a) {}
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const;
  float3 albedo;
};

class Metal : public Material {
public:
  __device__ Metal() {}
  __device__ Metal(const float3& a, float fuzz) : albedo(a), fuzz(fuzz) {}
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const;
  float3 albedo;
  float fuzz;
};

// class Dielectric : public Material {
// public:
//   __device__ Dielectric(float refIdx) : refIdx(refIdx) {}
//   __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
// 				  float3& attenuation, Ray& outRays,
// 				  curandState* localRandState) const;
//   float refIdx;
// };

// __device__ float schlick(float cosine, float refIdx);

#endif//MATERIALS_CUH
