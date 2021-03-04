#include "logging.h"
#include "GenericMaterial.h"

#include "vector_utils.h"

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
