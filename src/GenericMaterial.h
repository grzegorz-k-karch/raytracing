#ifndef GENERIC_MATERIAL_H
#define GENERIC_MATERIAL_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "StatusCodes.h"
#include "GenericMaterialDevice.cuh"
#include "GenericTexture.h"

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
  void parseDiffuseLight(const pt::ptree material);
  void parseLambertian(const pt::ptree material);
  void parseMetal(const pt::ptree material);
  void parseDielectric(const pt::ptree material);
  void parseParametric(const pt::ptree material);

  MaterialType m_materialType;
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
  std::vector<GenericTexture> m_textures;
};

#endif//GENERIC_MATERIAL_H
