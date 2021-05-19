#include <fstream>
#include <cmath>
#include <algorithm>
#include <climits>
#include <iostream>
#include <regex>
#include <pngwriter.h>

#include "nvidia/helper_math.h"
#include "logging.h"
#include "ImageSaver.h"


void ImageSaver::savePPM(const std::vector<float3>& image, int imageWidth, int imageHeight,
			 std::string pictureFilePath, StatusCodes& status) const
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
      ir = std::clamp(ir, 0, UCHAR_MAX);
      int ig = int(255.99f*color.y);
      ig = std::clamp(ig, 0, UCHAR_MAX);
      int ib = int(255.99f*color.z);
      ib = std::clamp(ib, 0, UCHAR_MAX);

      fs << ir << " " << ig << " " << ib << "\n";
    }
  }
  fs.close();
}

void ImageSaver::savePNG(const std::vector<float3>& image, int imageWidth, int imageHeight,
			 std::string pictureFilePath, StatusCodes& status) const
{
  pngwriter pngFile(imageWidth,imageHeight,0,pictureFilePath.c_str());
  for (int j = imageHeight - 1; j >= 0; j--) {
    for (int i = 0; i < imageWidth; i++) {
      size_t pixelIdx = i + j*imageWidth;
      float3 color = image[pixelIdx];
      // gamma correction
      color = make_float3(std::sqrt(color.x),
			  std::sqrt(color.y),
			  std::sqrt(color.z));
      color = clamp(color, 0.0f, 1.0f);
      pngFile.plot(i+1, j+1, color.x, color.y, color.z);
    }
  }
  pngFile.setcompressionlevel(0);
  pngFile.close();
}

void ImageSaver::saveImage(const std::vector<float3>& image, int imageWidth, int imageHeight,
			   std::string pictureFilePath, StatusCodes& status)
{
  if (m_imageType == ImageType::None) {
    std::smatch match_png;
    std::regex_match(pictureFilePath, match_png, std::regex(".*\.(png|PNG)"));
    std::smatch match_ppm;
    std::regex_match(pictureFilePath, match_ppm, std::regex(".*\.(ppm|PPM)"));

    LOG_TRIVIAL(info) << "Image file type inferred from name extension:";
    if (!(match_ppm.str(1).empty())) {
      LOG_TRIVIAL(info) << "\tFile type PPM.";
      m_imageType = ImageType::PPM;
    }
    else if (!(match_png.str(1).empty())) {
      LOG_TRIVIAL(info) << "\tFile type PNG.";
      m_imageType = ImageType::PNG;
    }
  }
  // with file type set, save the image
  if (m_imageType == ImageType::PPM) {
    savePPM(image, imageWidth, imageHeight,
	    pictureFilePath, status);
  }
  else if (m_imageType == ImageType::PNG) {
    savePNG(image, imageWidth, imageHeight,
	    pictureFilePath, status);
  }
  else {
    LOG_TRIVIAL(error) << "Image file type unknown. Saving \"" << pictureFilePath << "\" aborted.";
    status = StatusCodes::FileError;
    return;
  }
  LOG_TRIVIAL(info) << "Image saved as " << pictureFilePath << ".";
}
