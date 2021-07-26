#ifndef GENERIC_MATERIAL_H
#define GENERIC_MATERIAL_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "StatusCodes.h"
#include "GenericTexture.h"

namespace pt = boost::property_tree;

enum class MaterialType { None, DiffuseLight, Lambertian,
			  Metal, Dielectric, Parametric };

struct GenericMaterialDevice {

  GenericMaterialDevice() :
    m_materialType(MaterialType::None),
    m_scalars(nullptr), m_numScalars(0),
    m_vectors(nullptr), m_numVectors(0),
    m_textures(nullptr), m_numTextures(0) {}

  ~GenericMaterialDevice();

  MaterialType m_materialType;
  float *m_scalars;
  int m_numScalars;
  float3 *m_vectors;
  int m_numVectors;
  GenericTextureDevice *m_textures;
  int m_numTextures;
};


class GenericMaterial {
public:
  // default constructor
  GenericMaterial() = delete;
  // explicit constructor
  GenericMaterial(const pt::ptree& material, StatusCodes& status);
  // move constructor
  GenericMaterial(GenericMaterial&& other) noexcept;
  // copy constructor
  GenericMaterial(const GenericMaterial& other) = delete;
  // copy assignment operator
  GenericMaterial& operator=(const GenericMaterial& other) = delete;
  // move assignment operator
  GenericMaterial& operator=(const GenericMaterial&& other) = delete;

  void copyToDevice(GenericMaterialDevice* genericMaterialDevice,
		    StatusCodes& status);

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

  GenericMaterialDevice m_h_genericMaterialDevice;  
};

#endif//GENERIC_MATERIAL_H
