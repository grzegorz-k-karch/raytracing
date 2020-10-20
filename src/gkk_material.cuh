#ifndef GKK_MATERIAL_H
#define GKK_MATERIAL_H

#include "gkk_object.cuh"

#include <curand_kernel.h>

#include <boost/property_tree/ptree.hpp>
namespace pt = boost::property_tree;


class Material {
 public:
  __host__ static Material* create(pt::ptree tree);
  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const = 0;
  __host__ virtual std::size_t size() const = 0;
  __host__ virtual void print() const = 0;
  __device__ virtual void d_print() const = 0;
};

class Lambertian : public Material {
 public:
  __host__ __device__
  Lambertian(const vec3& a) : albedo(a) {}
  __host__ Lambertian(pt::ptree tree);
  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;
  __host__ virtual std::size_t size() const { return sizeof(Lambertian); };
  __host__ virtual void print() const;
  __device__ virtual void d_print() const;    

  vec3 albedo;
};

class Metal : public Material {
 public:
  __device__ Metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}
  __host__ Metal(pt::ptree tree);
  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;
  __host__ virtual std::size_t size() const { return sizeof(Metal); };
  __host__ virtual void print() const;
  __device__ virtual void d_print() const;
  
  vec3 albedo;
  float fuzz;
};

class Dielectric : public Material {
 public:
  __device__ Dielectric(float ref_idx) : ref_idx(ref_idx) {}
  __host__ Dielectric(pt::ptree tree);
  __device__ virtual bool scatter(const Ray& in_ray, const hit_record& hrec,
				  vec3& attenuation, Ray& out_rays,
				  curandState* local_rand_state) const;
  __host__ virtual std::size_t size() const { return sizeof(Dielectric); };
  __host__ virtual void print() const;
  __device__ virtual void d_print() const;    

  float ref_idx;
};

__device__ float schlick(float cosine, float ref_idx);

#endif//GKK_MATERIAL_H
