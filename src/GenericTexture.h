#ifndef GENERIC_TEXTURE_H
#define GENERIC_TEXTURE_H

#include <vector>
#include <vector_types.h>
#include <texture_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "StatusCodes.h"

namespace pt = boost::property_tree;

enum class TextureType { None, SolidColor, ImageTexture };

struct GenericTextureDevice {

  GenericTextureDevice() :
    m_textureType(TextureType::None),
    m_textureObject(0),
    m_vectors(nullptr),
    m_numVectors(0) {}

  ~GenericTextureDevice();

  TextureType m_textureType;
  // SolidColor
  float3 *m_vectors;
  int m_numVectors;
  // ImageTexture
  cudaTextureObject_t m_textureObject;
};


class GenericTexture {
public:
  GenericTexture(const pt::ptree& texture,
		 StatusCodes& status);
  void copyToDevice(GenericTextureDevice* d_genericTextureDevice,
		    StatusCodes& status);

private:
  void parseImageTexture(const pt::ptree& texture,
			 StatusCodes& status);
  void parseSolidColor(const pt::ptree& texture,
		       StatusCodes& status);
  void loadImageToDeviceTexture(cudaTextureObject_t& textureObject,
				StatusCodes& status);

  TextureType m_textureType;
  int m_imageWidth;
  int m_imageHeight;
  int m_numChannels;
  std::vector<float4> m_imageBuffer;
  std::vector<float3> m_vectors;

  GenericTextureDevice m_h_genericTextureDevice;
};

#endif//GENERIC_TEXTURE_H
