#include "gkk_material.cuh"
#include "gkk_random.cuh"

__device__ bool Lambertian::scatter(const Ray& in_ray, const hit_record& hrec,
				    vec3& attenuation, Ray& out_rays,
				    curandState* local_rand_state) const
{
  vec3 target = hrec.p + hrec.n + random_in_unit_sphere(local_rand_state);
  out_rays = Ray(hrec.p, target - hrec.p);
  attenuation = albedo;
  return true;
}


__device__ bool Metal::scatter(const Ray& in_ray, const hit_record& hrec,
			       vec3& attenuation, Ray& out_rays,
			       curandState* local_rand_state) const
{
  vec3 reflected = reflect(normalize(in_ray.direction()), hrec.n);
  out_rays = Ray(hrec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
  attenuation = albedo;
  return (dot(out_rays.direction(), hrec.n) > 0.0f);
}


__device__ bool Dielectric::scatter(const Ray& in_ray, const hit_record& hrec,
				    vec3& attenuation, Ray& out_rays,
				    curandState* local_rand_state) const
{
  vec3 outward_normal;
  float n1_over_n2;
  vec3 refracted;

  float reflect_prob;
  float cosine;

  attenuation = vec3(1.0f, 1.0f, 1.0f);

  if (dot(in_ray.direction(), hrec.n) > 0.0f) {
    outward_normal = -hrec.n;
    n1_over_n2 = ref_idx;
    cosine = ref_idx*dot(in_ray.direction(), hrec.n)/in_ray.direction().length();
  }
  else {
    outward_normal = hrec.n;
    n1_over_n2 = 1.0f/ref_idx;
    cosine = -dot(in_ray.direction(), hrec.n)/in_ray.direction().length();    
  }

  if (refract(in_ray.direction(), outward_normal, n1_over_n2, refracted)) {
    reflect_prob = schlick(cosine, ref_idx);
  }
  else {
    reflect_prob = 1.0f;
  }

  if (curand_uniform(local_rand_state) < reflect_prob) {
    vec3 reflected = reflect(normalize(in_ray.direction()), hrec.n);
    out_rays = Ray(hrec.p, reflected);
  }
  else {
    out_rays = Ray(hrec.p, refracted);
  }
  
  return true;  
}

__device__ float schlick(float cosine, float ref_idx)
{
  float r0 = (1.0f - ref_idx)/(1.0f + ref_idx);
  r0 = r0*r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}
