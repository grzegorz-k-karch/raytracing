#ifndef IMAGE_SAVER_H
#define IMAGE_SAVER_H

#include <string>
#include <vector>
#include "nvidia/helper_math.h"
#include "StatusCode.h"

enum class ImageType { None, PNG, PPM };

class ImageSaver {
public:
  ImageSaver(ImageType imageType=ImageType::None) :
    m_imageType(imageType) {}
  void saveImage(const std::vector<float3>& image,
		 int imageWidth, int imageHeight,
		 std::string pictureFilePath,
		 StatusCode& status);
private:
  void savePPM(const std::vector<float3>& image,
	       int imageWidth, int imageHeight,
	       std::string pictureFilePath,
	       StatusCode& status) const;  
  void savePNG(const std::vector<float3>& image,
	       int imageWidth, int imageHeight,
	       std::string pictureFilePath,
	       StatusCode& status) const;

  ImageType m_imageType;
};

#endif//IMAGE_SAVER_H
