#ifndef TEXTURES_CUH
#define TEXTURES_CUH


class Texture {
public:
  __device__ virtual float3 color(float u, float v,
				  const float3& p) const = 0;
};


class SolidColor : public Texture {
public:
  __device__ SolidColor() {}
  __device__ SolidColor(float3 color) : m_color(color) {}

  __device__ SolidColor(float red, float green, float blue)
    : SolidColor(make_float3(red,green,blue)) {}

  __device__ virtual float3 color(float u, float v,
				  const float3& p) const override {
    return m_color;
  }

private:
  float3 m_color;
};


class ImageTexture : public Texture {
public:
  __device__ ImageTexture(cudaTextureObject_t texObj) :
    m_image(texObj) {}
  __device__ virtual float3 color(float u, float v,
				  const float3& p) const override;
private:
  cudaTextureObject_t m_image;
};

#endif//TEXTURES_CUH
