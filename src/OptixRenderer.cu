#include <optix.h>

#include "nvidia/helper_math.h"

#include "OptixRenderer.cuh"

extern "C" {
  __constant__ Params params;
}


static __forceinline__ __device__ void setPayload(float3 p)
{
  optixSetPayload_0(float_as_int(p.x));
  optixSetPayload_1(float_as_int(p.y));
  optixSetPayload_2(float_as_int(p.z));
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
  const float2 d = 2.0f * make_float2(
				      static_cast<float>(idx.x) / static_cast<float>(dim.x),
				      static_cast<float>(idx.y) / static_cast<float>(dim.y)
				     ) - 1.0f;

  origin    = params.cam_eye;
  direction = normalize(make_float3(d.x, d.y, -1.0f));
}


extern "C" __global__ void __raygen__rg()
{
  // Lookup our location within the launch grid
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  // Map our launch idx to a screen location and create a ray from the camera
  // location through the screen
  float3 ray_origin, ray_direction;
  computeRay(idx, dim, ray_origin, ray_direction);

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
  params.image[idx.y * params.image_width + idx.x] = result;
}


extern "C" __global__ void __miss__ms()
{
  MissData* miss_data  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  setPayload( miss_data->bg_color);
}


extern "C" __global__ void __closesthit__ch()
{
  // When built-in triangle intersection is used, a number of fundamental
  // attributes are provided by the OptiX API, indlucing barycentric coordinates.
  const float2 barycentrics = optixGetTriangleBarycentrics();

  setPayload(make_float3(barycentrics, 1.0f));
}
