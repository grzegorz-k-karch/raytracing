#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include <assert.h>
#include "GenericMaterial.h"
#include "Ray.cuh"
#include "Textures.cuh"

class Material {
public:
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const = 0;
  __device__ virtual float3 emitted(float u, float v, const float3& p) const {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
};


class DiffuseLight : public Material  {
public:
  __device__ DiffuseLight(const GenericMaterialDevice *genMatDev);

  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
				  float3& attenuation, Ray& outRays,
				  curandState* localRandState) const override {
    return false;
  }

  __device__ virtual float3 emitted(float u, float v, const float3& p) const override {
    return m_emittingTexture->color(u, v, p);
  }

public:
  Texture *m_emittingTexture;
};

class Lambertian : public Material {
public:
  __device__ Lambertian(const GenericMaterialDevice *genMatDev);
  
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  Texture **m_textures;
  int m_numTextures;
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


class Parametric : public Material {
public:
  __device__ Parametric(const GenericMaterialDevice* genMatDev) :
    m_dummy(genMatDev->scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float m_dummy;
};

__device__ float schlick(float cosine, float refIdx);

#endif//MATERIALS_CUH
