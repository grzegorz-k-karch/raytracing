#include "Textures.cuh"

__device__ float3 ImageTexture::color(float u, float v,
				      const float3& p) const
{
  float4 texValue = tex2D<float4>(m_image, u, v);
  return make_float3(texValue.x, texValue.y, texValue.z);
}
