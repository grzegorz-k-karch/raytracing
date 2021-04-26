#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include <assert.h>
#include "GenericMaterial.h"
#include "Ray.cuh"

class Material {
public:
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const = 0;
};

class Lambertian : public Material {
public:
  __device__ Lambertian(const GenericMaterialDevice* genMatDev)
    : m_albedo(genMatDev->vectors[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float3 m_albedo;
};

class Metal : public Material {
public:
  __device__ Metal(const GenericMaterialDevice* genMatDev)
    : m_albedo(genMatDev->vectors[0]),
      m_fuzz(genMatDev->scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float3 m_albedo;
  float m_fuzz;
};

class Dielectric : public Material {
public:
  __device__ Dielectric(const GenericMaterialDevice* genMatDev)
    : m_refIdx(genMatDev->scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float m_refIdx;
};

__device__ float schlick(float cosine, float refIdx);

#endif//MATERIALS_CUH
