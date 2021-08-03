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
  __device__ Lambertian() = delete;
  __device__ Lambertian(const GenericMaterialDevice *genMatDev);
  __device__ ~Lambertian();  

  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  Texture **m_textures;
  int m_numTextures;
};


class Metal : public Material {
public:
  __device__ Metal(const GenericMaterialDevice* genMatDev)
    : m_albedo(genMatDev->m_vectors[0]),
      m_fuzz(genMatDev->m_scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float3 m_albedo;
  float m_fuzz;
};


class Dielectric : public Material {
public:
  __device__ Dielectric(const GenericMaterialDevice* genMatDev)
    : m_refIdx(genMatDev->m_scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float m_refIdx;
};


class Parametric : public Material {
public:
  __device__ Parametric(const GenericMaterialDevice* genMatDev) :
    m_dummy(genMatDev->m_scalars[0]) {}
  __device__ virtual bool scatter(const Ray& inRay, const HitRecord& hitRec,
  				  float3& attenuation, Ray& outRays,
  				  curandState* localRandState) const;
  float m_dummy;
};

__device__ float schlick(float cosine, float refIdx);


class MaterialFactory {
public:
  __device__
  static Material* createMaterial(const GenericMaterialDevice* genMatDev) {

    Material *material = nullptr;
    switch (genMatDev->m_materialType) {
    case MaterialType::DiffuseLight:
      material = new DiffuseLight(genMatDev);
      break;
    case MaterialType::Lambertian:
      material = new Lambertian(genMatDev);
      break;
    case MaterialType::Metal:
      material = new Metal(genMatDev);
      break;
    case MaterialType::Dielectric:
      material = new Dielectric(genMatDev);
      break;
    case MaterialType::Parametric:
      material = new Parametric(genMatDev);
      break;
    case MaterialType::None:
      break;
    default:
      break;
    }
    assert(material != nullptr);
    return material;
  }
};

#endif//MATERIALS_CUH
