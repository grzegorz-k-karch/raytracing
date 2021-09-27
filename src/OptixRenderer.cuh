#ifndef OPTIX_RENDERER_CUH
#define OPTIX_RENDERER_CUH

struct Params
{
    float3*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
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
