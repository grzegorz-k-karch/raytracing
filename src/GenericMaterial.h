#ifndef GENERIC_MATERIAL_H
#define GENERIC_MATERIAL_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "logging.h"
#include "StatusCodes.h"
#include "GenericMaterialDevice.cuh"

namespace pt = boost::property_tree;

class GenericMaterial {
public:

  GenericMaterial(const std::string materialType, const pt::ptree material) {
    if (materialType == "Lambertian") {
      m_materialType = MaterialType::Lambertian;
      parseLambertian(material);      
    }
    else if (materialType  == "Metal") {
      m_materialType = MaterialType::Metal;
      parseMetal(material);      
    }
  }

  void copyToDevice(GenericMaterialDevice* genericMaterialDevice,
		    StatusCodes& status);

private:
  void parseLambertian(const pt::ptree material);
  void parseMetal(const pt::ptree material);

  MaterialType m_materialType;
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
};

#endif//GENERIC_MATERIAL_H
