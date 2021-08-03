#include "logging.h"
#include "GenericMaterial.h"
#include "nvidia/helper_math.h"
#include "vector_utils.cuh"

GenericMaterial::GenericMaterial(const pt::ptree& material,
				 StatusCode& status)
{
  std::string materialType = material.get<std::string>("<xmlattr>.value");
  if (materialType == "DiffuseLight") {
    m_materialType = MaterialType::DiffuseLight;
    parseDiffuseLight(material);
  }
  else if (materialType == "Lambertian") {
    m_materialType = MaterialType::Lambertian;
    parseLambertian(material);
  }
  else if (materialType  == "Metal") {
    m_materialType = MaterialType::Metal;
    parseMetal(material);
  }
  else if (materialType  == "Dielectric") {
    m_materialType = MaterialType::Dielectric;
    parseDielectric(material);
  }
  else if (materialType  == "Parametric") {
    m_materialType = MaterialType::Parametric;
    parseParametric(material);
  }
}

GenericMaterial::GenericMaterial(GenericMaterial&& other) noexcept :
  m_materialType(other.m_materialType),
  m_scalars(other.m_scalars),
  m_vectors(other.m_vectors),
  m_textures(other.m_textures)
{
  LOG_TRIVIAL(trace) << "GenericMaterial copy constructor";  
}


void GenericMaterial::parseDiffuseLight(const pt::ptree material)
{
  StatusCode status = StatusCode::NoError;
  //--------------------------------------------------------------------------
  // texture
  auto texture_it = material.find("texture");
  bool textureFound = texture_it != material.not_found();
  if (textureFound) {
    LOG_TRIVIAL(trace) << "Texture found.";
    pt::ptree texture = texture_it->second;
    m_textures.push_back(GenericTexture(texture, status));
  }
  else {
    LOG_TRIVIAL(trace) << "Texture not found.";
  }
}


void GenericMaterial::parseLambertian(const pt::ptree material)
{
  StatusCode status = StatusCode::NoError;
  //--------------------------------------------------------------------------
  // texture
  auto texture_it = material.find("texture");
  bool textureFound = texture_it != material.not_found();
  if (textureFound) {
    LOG_TRIVIAL(trace) << "Texture found.";
    pt::ptree texture = texture_it->second;
    m_textures.push_back(GenericTexture(texture, status));
  }
  else {
    LOG_TRIVIAL(trace) << "Texture not found.";
  }
}


void GenericMaterial::parseMetal(const pt::ptree material)
{
  float3 albedo = string2float3(material.get<std::string>("albedo.<xmlattr>.value"));
  m_vectors = {albedo};
  float fuzz = material.get<float>("fuzz.<xmlattr>.value");
  m_scalars = {fuzz};
}


void GenericMaterial::parseDielectric(const pt::ptree material)
{
  float refIdx = material.get<float>("ref_idx.<xmlattr>.value");
  m_scalars = {refIdx};
}

void GenericMaterial::parseParametric(const pt::ptree material)
{
  m_scalars = {0.0f};
}
