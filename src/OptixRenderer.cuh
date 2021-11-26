#ifndef OPTIX_RENDERER_CUH
#define OPTIX_RENDERER_CUH

#include "Camera.cuh"

struct Params
{
  float3*                image;
  unsigned int           image_width;
  unsigned int           image_height;
  OptixTraversableHandle handle;
  Camera*                camera;  
};


struct RayGenData
{
  // No data needed
};


struct MissData
{
  float3 bg_color;
};


struct HitGroupData
{
  // No data needed
};

#endif //OPTIX_RENDERER_CUH
