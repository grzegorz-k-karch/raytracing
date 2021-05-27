#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

#include "logging.h"

enum class MaterialType { None, Lambertian, Metal, Dielectric, Parametric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    materialType(MaterialType::None),
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0) {}

  ~GenericMaterialDevice() {}

  MaterialType materialType;

  float  *scalars;
  int numScalars;

  float3 *vectors;
  int numVectors;
};

#endif//GENERIC_MATERIAL_DEVICE_H
