#include <vector_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"

#include "logging.h"

#include "TextureImageLoader.h"


TextureImageLoader::TextureImageLoader(const pt::ptree& texture)
{
  m_textureFilepath = texture.get<std::string>("source.<xmlattr>.value");
  LOG_TRIVIAL(debug) << "Texture filepath: " << m_textureFilepath;
}

void TextureImageLoader::loadImage(int &imageWidth, int &imageHeight,
				   int &numChannels,
				   std::vector<float4> &imageBuffer,
				   StatusCode& status)
{
  status = StatusCode::NoError;
  const int desiredNumChannels = 3;
  unsigned char *data = stbi_load(m_textureFilepath.c_str(), &imageWidth, &imageHeight,
				  &numChannels, desiredNumChannels);
  if (desiredNumChannels != numChannels) {
    LOG_TRIVIAL(warning) << "In " << m_textureFilepath
			 << " the number of channels = " << numChannels
			 << " is different than the desired number of channels = "
			 << desiredNumChannels << ".";
  }
  int numImagePixels = imageWidth*imageHeight;
  imageBuffer.resize(numImagePixels);
  int componentIdx = 0;
  for (int pixelIdx = 0; pixelIdx < numImagePixels; pixelIdx++) {
    float pixelBuffer[3];
    for (int channelIdx = 0; channelIdx < desiredNumChannels; channelIdx++) {
      pixelBuffer[channelIdx] = data[componentIdx]/255.0f;
      componentIdx++;
    }
    imageBuffer[pixelIdx] = make_float4(pixelBuffer[0],
					pixelBuffer[1],
					pixelBuffer[2], 0.0f);
  }
}
