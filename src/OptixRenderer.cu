#include <optix.h>

#include "nvidia/helper_math.h"

#include "OptixRenderer.cuh"

extern "C" {
  __constant__ Params params;
}


__forceinline__ __device__ float3 toSRGB(const float3& c)
{
  float  invGamma = 1.0f / 2.4f;
  float3 powed    = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
  return make_float3(c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
		     c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
		     c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}


__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
  x = clamp(x, 0.0f, 1.0f);
  enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
  return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}


__forceinline__ __device__ uchar4 make_color(const float3& c)
{
  // first apply gamma, then convert to unsigned char
  float3 srgb = toSRGB(clamp(c, 0.0f, 1.0f));
  return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}


static __forceinline__ __device__ void setPayload(float3 p)
{
  optixSetPayload_0(float_as_int(p.x));
  optixSetPayload_1(float_as_int(p.y));
  optixSetPayload_2(float_as_int(p.z));
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
  const float3 U = params.cam_u;
  const float3 V = params.cam_v;
  const float3 W = params.cam_w;
  const float2 d = 2.0f * make_float2(
				      static_cast<float>(idx.x) / static_cast<float>(dim.x),
				      static_cast<float>(idx.y) / static_cast<float>(dim.y)
				      ) - 1.0f;

  origin    = params.cam_eye;
  direction = normalize(d.x * U + d.y * V + W);
}


extern "C" __global__ void __raygen__renderScene()
{
  // Lookup our location within the launch grid
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  float3 ray_origin, ray_direction;
  computeRay(idx, dim, ray_origin, ray_direction);
  // float u = float(idx.x + curand_uniform(&localRandState))/float(dim.x);
  // float v = float(idx.y + curand_uniform(&localRandState))/float(dim.y);
  // Ray ray = camera->getRay(u, v, &localRandState);
      
  // Trace the ray against our scene hierarchy
  unsigned int p0, p1, p2;
  optixTrace(
	     params.handle,
	     ray_origin,
	     ray_direction,
	     0.0f,                // Min intersection distance
	     1e16f,               // Max intersection distance
	     0.0f,                // rayTime -- used for motion blur
	     OptixVisibilityMask(255), // Specify always visible
	     OPTIX_RAY_FLAG_NONE,
	     0,                   // SBT offset   -- See SBT discussion
	     1,                   // SBT stride   -- See SBT discussion
	     0,                   // missSBTIndex -- See SBT discussion
	     p0, p1, p2);
  float3 result;
  result.x = int_as_float(p0);
  result.y = int_as_float(p1);
  result.z = int_as_float(p2);

  // Record results in our output raster
  params.image[idx.y * params.image_width + idx.x] = make_color(result);
}


extern "C" __global__ void __miss__ms()
{
  MissData* miss_data  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  setPayload(miss_data->bg_color);
}


extern "C" __global__ void __closesthit__ch()
{
  // When built-in triangle intersection is used, a number of fundamental
  // attributes are provided by the OptiX API, indlucing barycentric coordinates.
  const float2 barycentrics = optixGetTriangleBarycentrics();

  setPayload(make_float3(barycentrics, 1.0f));
}