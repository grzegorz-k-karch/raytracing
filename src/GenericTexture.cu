#include "logging.h"
#include "cuda_utils.cuh"
#include "vector_utils.cuh"
#include "TextureImageLoader.h"
#include "GenericTexture.h"

void GenericTexture::copyToDevice(GenericTextureDevice* genericTextureDevice,
				  StatusCodes& status) const
{
  status = StatusCodes::NoError;

  GenericTextureDevice h_genericTextureDevice;

  h_genericTextureDevice.m_numVectors = m_vectors.size();
  h_genericTextureDevice.m_textureType = m_textureType;
  h_genericTextureDevice.m_textureObject = m_textureObject;

  // vectors
  int dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericTextureDevice.m_vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericTextureDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // whole texture
  status = CCE(cudaMemcpy(genericTextureDevice, &h_genericTextureDevice,
			  sizeof(GenericTextureDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}


GenericTexture::GenericTexture(const pt::ptree& texture,
			       StatusCodes& status)
{
  std::string textureType = texture.get<std::string>("<xmlattr>.value");
  if (textureType == "SolidColor") {
    LOG_TRIVIAL(trace) << "Solid color texture.";
    m_textureType = TextureType::SolidColor;
    parseSolidColor(texture, status);
  }
  else if (textureType  == "ImageTexture") {
    LOG_TRIVIAL(trace) << "Image texture.";
    m_textureType = TextureType::ImageTexture;
    parseImageTexture(texture, status);
  }
}


void GenericTexture::parseSolidColor(const pt::ptree& texture,
				     StatusCodes& status)
{
  float3 albedo = string2float3(texture.get<std::string>("albedo.<xmlattr>.value"));
  m_vectors = {albedo};
}


void GenericTexture::parseImageTexture(const pt::ptree& texture,
				       StatusCodes& status)
{
  status = StatusCodes::NoError;
  int imageWidth;
  int imageHeight;
  int numChannels;
  std::vector<float4> h_imageBuffer;
  TextureImageLoader textureImageLoader(texture);
  textureImageLoader.loadImage(imageWidth, imageHeight, numChannels,
  			       h_imageBuffer, status);

  // create cuda texture
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray_t cuArray;
  status = CCE(cudaMallocArray(&cuArray, &channelDesc,
			       imageWidth, imageHeight));
  if (status != StatusCodes::NoError) {
    return;
  }

  // Set pitch of the source (the width in memory in bytes
  // of the 2D array pointed to by src, including padding)
  const size_t spitch = imageWidth*sizeof(float4);
  // Copy data located at address h_data in host memory to device memory
  status = CCE(cudaMemcpy2DToArray(cuArray, 0, 0, h_imageBuffer.data(), spitch,
				   imageWidth*sizeof(float4), imageHeight,
				   cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  m_textureObject = 0;
  status = CCE(cudaCreateTextureObject(&m_textureObject, &resDesc,
				       &texDesc, NULL));
  if (status != StatusCodes::NoError) {
    return;
  }
  if (m_textureObject == 0) {
    LOG_TRIVIAL(error) << "Could not create texture object.";
    status = StatusCodes::CudaError;
    return;
  }
}
