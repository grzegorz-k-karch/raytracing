#include "ImageSaver.h"
#include <fstream>
#include <cmath>

void ImageSaver::saveImage(const std::vector<float3>& image,
			   int imageWidth, int imageHeight,
			   std::string pictureFilePath,
			   StatusCodes& status) const
{
  std::fstream fs(pictureFilePath, std::fstream::out);
  
  fs << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";
  
  for (int j = imageHeight - 1; j >= 0; j--) {
    for (int i = 0; i < imageWidth; i++) {
      
      size_t pixelIdx = i + j*imageWidth;
      float3 color = image[pixelIdx];
      // gamma correction
      color = make_float3(std::sqrt(color.x),
			  std::sqrt(color.y),
			  std::sqrt(color.z));
      
      int ir = int(255.99f*color.x);
      int ig = int(255.99f*color.y);
      int ib = int(255.99f*color.z);
      
      fs << ir << " " << ig << " " << ib << "\n";
    }
  }
  fs.close();
}

