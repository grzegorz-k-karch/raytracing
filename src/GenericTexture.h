#ifndef GENERIC_TEXTURE_H
#define GENERIC_TEXTURE_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "StatusCodes.h"
#include "GenericTextureDevice.cuh"

namespace pt = boost::property_tree;

class GenericTexture {
public:
  GenericTexture(const std::string& textureType,
		 const pt::ptree texture, StatusCodes& status);
  void copyToDevice(GenericTextureDevice* genericTextureDevice,
		    StatusCodes& status) const;

private:
  void parseImageTexture(const pt::ptree texture, StatusCodes& status);
  void parseSolidColor(const pt::ptree texture, StatusCodes& status);

  TextureType m_textureType;
  std::vector<float3> m_vectors;
  cudaTextureObject_t m_textureObject;  
};

#endif//GENERIC_TEXTURE_H
