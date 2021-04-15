#ifndef GENERIC_MATERIAL_DEVICE_H
#define GENERIC_MATERIAL_DEVICE_H

#include "logging.h"

// #include "cuda_utils.cuh"

enum class MaterialType { None, Lambertian, Metal, Dielectric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0) {
#if __CUDA_ARCH__ >= 200
    printf("GenericMaterialDevice constructor on device\n");
#else
    LOG_TRIVIAL(trace) << "GenericMaterialDevice constructor on host";
#endif
    
  }

  ~GenericMaterialDevice() {
#if __CUDA_ARCH__ >= 200
    printf("GenericMaterialDevice destructor on device\n");
#else
    LOG_TRIVIAL(trace) << "GenericMaterialDevice destructor on host";
#endif
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
