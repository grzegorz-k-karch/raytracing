#ifndef GENERIC_TEXTURE_DEVICE_H
#define GENERIC_TEXTURE_DEVICE_H

#include <texture_types.h>

#include "logging.h"

enum class TextureType { None, SolidColor, ImageTexture };

struct GenericTextureDevice {

  GenericTextureDevice() :
    m_textureType(TextureType::None),
    m_textureObject(0),
    m_vectors(nullptr),
    m_numVectors(0) {}

  ~GenericTextureDevice() {}

  TextureType m_textureType;
  // SolidColor
  float3 *m_vectors;
  int m_numVectors;
  // ImageTexture
  cudaTextureObject_t m_textureObject;
};

#endif//GENERIC_TEXTURE_DEVICE_H
