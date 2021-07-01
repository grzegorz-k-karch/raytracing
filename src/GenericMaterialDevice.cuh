#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

#include "logging.h"
#include "GenericTextureDevice.cuh"

enum class MaterialType { None, DiffuseLight, Lambertian, Metal, Dielectric, Parametric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    m_materialType(MaterialType::None),
    m_scalars(nullptr), m_numScalars(0),
    m_vectors(nullptr), m_numVectors(0),
    m_textures(nullptr), m_numTextures(0) {}

  ~GenericMaterialDevice() {}

  MaterialType m_materialType;
  float *m_scalars;
  int m_numScalars;
  float3 *m_vectors;
  int m_numVectors;
  GenericTextureDevice *m_textures;
  int m_numTextures;
};

#endif//GENERIC_MATERIAL_DEVICE_H
