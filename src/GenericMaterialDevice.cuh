#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

enum class MaterialType { None, Lambertian, Metal, Dielectric };

struct GenericMaterialDevice {

  MaterialType materialType;

  float  *scalars;
  int numScalars;

  float3 *vectors;
  int numVectors;
};

#endif//GENERIC_MATERIAL_DEVICE_H
