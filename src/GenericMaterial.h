#ifndef GENERIC_MATERIAL_H
#define GENERIC_MATERIAL_H

#include <vector_types.h>

enum class MaterialType { None, Lambertian, Metal };

class GenericMaterial {
 public:

  GenericMaterial(const std::string materialType, const pt::ptree material) {
    if (materialType == "Lambertian") {
      m_materialType = MaterialType::Lambertian;
    }
    else if (materialType  == "Metal") {
      m_materialType = MaterialType::Metal;
    }
  }

  MaterialType m_materialType;
  int m_numScalars;
  float *m_scalars;
  int m_numVectors;
  float3 *m_vectors;
};

#endif//GENERIC_MATERIAL_H
