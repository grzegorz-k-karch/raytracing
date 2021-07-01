#include "Materials.cuh"
#include "Textures.cuh"
#include "vector_utils.cuh"
#include "nvidia/helper_math.h"


__device__ DiffuseLight::DiffuseLight(const GenericMaterialDevice *genMatDev)
{
  m_emittingTexture = TextureFactory::createTexture(&(genMatDev->m_textures[0]));
}


__device__ Lambertian::Lambertian(const GenericMaterialDevice *genMatDev)
{
  m_numTextures = genMatDev->m_numTextures;
  m_textures = new Texture*[m_numTextures];
  for (int texIdx = 0; texIdx < m_numTextures; texIdx++) {
    m_textures[texIdx] = TextureFactory::createTexture(&(genMatDev->m_textures[texIdx]));
  }
}


__device__ bool Lambertian::scatter(const Ray& inRay, const HitRecord& hitRec,
				    float3& attenuation, Ray& outRays,
				    curandState* localRandState) const
{
  float3 target = hitRec.p + hitRec.n + randomInUnitSphere(localRandState);
  outRays = Ray(hitRec.p, target - hitRec.p);
  attenuation = make_float3(1.0f, 1.0f, 1.0f);
  for (int texIdx = 0; texIdx < m_numTextures; texIdx++) {
    attenuation *= m_textures[texIdx]->color(hitRec.u, hitRec.v, hitRec.p);
  }
  return true;
}

__device__ bool Metal::scatter(const Ray& inRay, const HitRecord& hitRec,
			       float3& attenuation, Ray& outRays,
			       curandState* localRandState) const
{
  float3 reflected = reflect(normalize(inRay.m_direction), hitRec.n);
  if (dot(reflected, hitRec.n) < 0.0f) {
    reflected = -1.0f*reflected;
  }
  outRays = Ray(hitRec.p, reflected +
		m_fuzz*randomInUnitSphere(localRandState));
  attenuation = m_albedo;
  return (dot(outRays.m_direction, hitRec.n) > 0.0f);
}

__device__ bool Dielectric::scatter(const Ray& inRay, const HitRecord& hitRec,
				    float3& attenuation, Ray& outRays,
				    curandState* localRandState) const
{
  float3 outwardNormal;
  float n1OverN2;
  float3 refracted;

  float reflectProb;
  float cosine;

  attenuation = make_float3(1.0f, 1.0f, 1.0f);

  if (dot(inRay.m_direction, hitRec.n) > 0.0f) {
    outwardNormal = -hitRec.n;
    n1OverN2 = m_refIdx;
    cosine = m_refIdx*dot(inRay.m_direction,
			  hitRec.n)/length(inRay.m_direction);
  }
  else {
    outwardNormal = hitRec.n;
    n1OverN2 = 1.0f/m_refIdx;
    cosine = -dot(inRay.m_direction, hitRec.n)/length(inRay.m_direction);
  }

  if (refract(inRay.m_direction, outwardNormal, n1OverN2, refracted)) {
    reflectProb = schlick(cosine, m_refIdx);
  }
  else {
    reflectProb = 1.0f;
  }

  if (curand_uniform(localRandState) < reflectProb) {
    float3 reflected = reflect(normalize(inRay.m_direction), hitRec.n);
    outRays = Ray(hitRec.p, reflected);
  }
  else {
    outRays = Ray(hitRec.p, refracted);
  }

  return true;
}

__device__ float schlick(float cosine, float refIdx)
{
  float r0 = (1.0f - refIdx)/(1.0f + refIdx);
  r0 = r0*r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}


__device__ bool Parametric::scatter(const Ray& inRay, const HitRecord& hitRec,
				    float3& attenuation, Ray& outRays,
				    curandState* localRandState) const
{
  float3 target = hitRec.p + hitRec.n + randomInUnitSphere(localRandState);
  outRays = Ray(hitRec.p, target - hitRec.p);
  attenuation = fabs(hitRec.n);
  return true;
}
