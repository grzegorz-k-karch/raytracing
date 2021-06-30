#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

#include "logging.h"
#include "GenericTextureDevice.cuh"

enum class MaterialType { None, DiffuseLight, Lambertian, Metal, Dielectric, Parametric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    materialType(MaterialType::None),
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0),
    textures(nullptr), numTextures(0) {}

  ~GenericMaterialDevice() {}

  MaterialType materialType;

  float  *scalars;
  int numScalars;

  float3 *vectors;
  int numVectors;

  GenericTextureDevice *textures;
  int numTextures;
};

#endif//GENERIC_MATERIAL_DEVICE_H
