#ifndef TEXTURE_IMAGE_LOADER_H
#define TEXTURE_IMAGE_LOADER_H

#include <string>
#include <vector>
#include <boost/property_tree/xml_parser.hpp>

#include <vector_types.h>

#include "StatusCodes.h"

namespace pt = boost::property_tree;

class TextureImageLoader {
 public:
  TextureImageLoader(const pt::ptree& texture);
  void loadImage(int& imageWidth, int& imageHeight,
		 int& numChannels, std::vector<float4>& imageBuffer,
		 StatusCodes& status);
 private:
  std::string m_textureFilepath;
};

#endif//TEXTURE_IMAGE_LOADER_H
