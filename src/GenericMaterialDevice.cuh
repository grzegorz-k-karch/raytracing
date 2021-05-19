#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

#include "logging.h"

// #include "cuda_utils.cuh"

enum class MaterialType { None, Lambertian, Metal, Dielectric, Parametric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0) {}

  ~GenericMaterialDevice() {
    // if (scalars != nullptr) {
    //   CCE(cudaFree(scalars));
    // }
    // if (vectors != nullptr) {
    //   CCE(cudaFree(vectors));
    // }    
  }

  MaterialType materialType;

  float  *scalars;
  int numScalars;

  float3 *vectors;
  int numVectors;
};

#endif//GENERIC_MATERIAL_DEVICE_H
