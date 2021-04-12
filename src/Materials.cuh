#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include <assert.h>

#include "GenericMaterial.h"

class Material {
public:
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const = 0;
};

class Lambertian : public Material {
public:
  __device__ Lambertian(const GenericMaterialDevice* genMatDev)
    : albedo(genMatDev->vectors[0]) {}
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const;
  float3 albedo;
};

class Metal : public Material {
public:
  __device__ Metal(const GenericMaterialDevice* genMatDev)
    : albedo(genMatDev->vectors[0]),
      fuzz(genMatDev->scalars[0]) {}
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const;
  float3 albedo;
  float fuzz;
};

class Dielectric : public Material {
public:
  __device__ Dielectric(const GenericMaterialDevice* genMatDev)
    : refIdx(genMatDev->scalars[0]) {}
  // __device__ virtual bool scatter(const Ray& inRay, const hit_record& hitRec,
  // 				  float3& attenuation, Ray& outRays,
  // 				  curandState* localRandState) const;
  float refIdx;
};

// __device__ float schlick(float cosine, float refIdx);

#endif//MATERIALS_CUH
