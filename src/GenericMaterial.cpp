#include "logging.h"
#include "GenericMaterial.h"
#include "nvidia/helper_math.h"
#include "vector_utils.cuh"

GenericMaterial::GenericMaterial(const std::string materialType,
				 const pt::ptree material)
{
  if (materialType == "Lambertian") {
    m_materialType = MaterialType::Lambertian;
    parseLambertian(material);
  }
  else if (materialType  == "Metal") {
    m_materialType = MaterialType::Metal;
    parseMetal(material);
  }
}

GenericMaterial::GenericMaterial(GenericMaterial&& other) noexcept :
  m_materialType(other.m_materialType),
  m_scalars(other.m_scalars),
  m_vectors(other.m_vectors)
{
}

void GenericMaterial::parseLambertian(const pt::ptree material)
{
  float3 albedo = string2float3(material.get<std::string>("albedo.<xmlattr>.value"));
  m_vectors = {albedo};
}

void GenericMaterial::parseMetal(const pt::ptree material)
{
  float3 albedo = string2float3(material.get<std::string>("albedo.<xmlattr>.value"));
  m_vectors = {albedo};
  float fuzz = material.get<float>("fuzz.<xmlattr>.value");
  m_scalars = {fuzz};
}
