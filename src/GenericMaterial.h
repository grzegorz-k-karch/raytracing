#ifndef GENERIC_MATERIAL_H
#define GENERIC_MATERIAL_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "StatusCodes.h"
#include "GenericMaterialDevice.cuh"

namespace pt = boost::property_tree;

class GenericMaterial {
public:
  // default constructor
  GenericMaterial() = delete;
  // explicit constructor
  GenericMaterial(const std::string materialType, const pt::ptree material);
  // move constructor
  GenericMaterial(GenericMaterial&& other) noexcept;
  // copy constructor
  GenericMaterial(const GenericMaterial& other) = delete;
  // copy assignment operator
  GenericMaterial& operator=(const GenericMaterial& other) = delete;
  // move assignment operator
  GenericMaterial& operator=(const GenericMaterial&& other) = delete;

  void copyToDevice(GenericMaterialDevice* genericMaterialDevice,
		    StatusCodes& status) const;

private:
  void parseLambertian(const pt::ptree material);
  void parseMetal(const pt::ptree material);
  void parseDielectric(const pt::ptree material);  

  MaterialType m_materialType;
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
};

#endif//GENERIC_MATERIAL_H
