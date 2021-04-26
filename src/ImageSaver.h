#ifndef IMAGE_SAVER_H
#define IMAGE_SAVER_H

#include <string>
#include <vector>
#include "nvidia/helper_math.h"
#include "StatusCodes.h"

class ImageSaver {
 public:
  void saveImage(const std::vector<float3>& image,
		 int imageWidth, int imageHeight,
		 std::string pictureFilePath,
		 StatusCodes& status) const;
};

#endif//IMAGE_SAVER_H
