#include "Materials.cuh"
#include "vector_utils.cuh"
#include "nvidia/helper_math.h"

__device__ bool Lambertian::scatter(const Ray& inRay, const HitRecord& hitRec,
				    float3& attenuation, Ray& outRays,
				    curandState* localRandState) const
{
  float3 target = hitRec.p + hitRec.n + randomInUnitSphere(localRandState);
  outRays = Ray(hitRec.p, target - hitRec.p, inRay.m_timestamp);
  attenuation = m_albedo;
  return true;
}

__device__ bool Metal::scatter(const Ray& inRay, const HitRecord& hitRec,
			       float3& attenuation, Ray& outRays,
			       curandState* localRandState) const
{
  float3 reflected = reflect(normalize(inRay.m_direction), hitRec.n);
  outRays = Ray(hitRec.p, reflected +
		 m_fuzz*randomInUnitSphere(localRandState), inRay.m_timestamp);
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
    outRays = Ray(hitRec.p, reflected, inRay.m_timestamp);
  }
  else {
    outRays = Ray(hitRec.p, refracted, inRay.m_timestamp);
  }
  
  return true;  
}

__device__ float schlick(float cosine, float refIdx)
{
  float r0 = (1.0f - refIdx)/(1.0f + refIdx);
  r0 = r0*r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}
