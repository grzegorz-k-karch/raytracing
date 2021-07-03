#ifndef TEXTURES_CUH
#define TEXTURES_CUH

#include "GenericTexture.h"

class Texture {
public:
  __device__ virtual float3 color(float u, float v,
				  const float3& p) const = 0;
};


class SolidColor : public Texture {
public:
  __device__ SolidColor(const GenericTextureDevice *genTexDev) :
    m_color(genTexDev->m_vectors[0]) {}

  __device__ virtual float3 color(float u, float v,
				  const float3& p) const override {
    return m_color;
  }

private:
  float3 m_color;
};


class ImageTexture : public Texture {
public:
  __device__ ImageTexture(const GenericTextureDevice *genTexDev) :
    m_image(genTexDev->m_textureObject) {}
  __device__ virtual float3 color(float u, float v,
				  const float3& p) const override;
private:
  cudaTextureObject_t m_image;
};


class TextureFactory {
public:
  __device__
  static Texture *createTexture(const GenericTextureDevice *genTexDev) {

    Texture *tex = nullptr;
    switch (genTexDev->m_textureType) {
    case TextureType::SolidColor:
      tex = new SolidColor(genTexDev);
      break;
    case TextureType::ImageTexture:
      tex = new ImageTexture(genTexDev);
      break;
    default:
      break;
    }
    assert(tex != nullptr);
    return tex;
  }
};

#endif//TEXTURES_CUH
