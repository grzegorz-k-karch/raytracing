#include "logging.h"
#include "cuda_utils.cuh"
#include "vector_utils.cuh"
#include "TextureImageLoader.h"
#include "GenericTexture.h"

GenericTextureDevice::~GenericTextureDevice()
{
  LOG_TRIVIAL(trace) << "~GenericTextureDevice";
  m_textureType = TextureType::None;
  if (m_vectors) {
    CCE(cudaFree(m_vectors));
    m_vectors = nullptr;
  }
  m_numVectors = 0;

  if (m_textureObject) {
    // retrieve cuArray hodling texture buffer
    cudaResourceDesc pResDesc;
    CCE(cudaGetTextureObjectResourceDesc(&pResDesc, m_textureObject));
    CCE(cudaFreeArray(pResDesc.res.array.array));
  
    CCE(cudaDestroyTextureObject(m_textureObject));
    m_textureObject = 0;
  }
}


void GenericTexture::loadImageToDeviceTexture(cudaTextureObject_t& textureObject,
					      StatusCode& status)
{
  // create cuda texture
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray_t cuArray;
  LOG_TRIVIAL(trace) << "m_imageWidth = " << m_imageWidth
		     << " m_imageHeight = " << m_imageHeight;  
  status = CCE(cudaMallocArray(&cuArray, &channelDesc,
			       m_imageWidth, m_imageHeight));
  if (status != StatusCode::NoError) {
    return;
  }

  // Set pitch of the source (the width in memory in bytes
  // of the 2D array pointed to by src, including padding)
  const size_t spitch = m_imageWidth*sizeof(float4);
  // Copy data located at address h_data in host memory to device memory
  status = CCE(cudaMemcpy2DToArray(cuArray, 0, 0, m_imageBuffer.data(), spitch,
				   m_imageWidth*sizeof(float4), m_imageHeight,
				   cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
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
  textureObject = 0;
  status = CCE(cudaCreateTextureObject(&(textureObject), &resDesc,
				       &texDesc, NULL));
  if (status != StatusCode::NoError) {
    return;
  }
  if (textureObject == 0) {
    LOG_TRIVIAL(error) << "Could not create texture object.";
    status = StatusCode::CudaError;
    return;
  }
}


void GenericTexture::copyToDevice(GenericTextureDevice* d_genericTextureDevice,
				  StatusCode& status)
{
  status = StatusCode::NoError;

  m_h_genericTextureDevice.m_textureType = m_textureType;
  m_h_genericTextureDevice.m_numVectors = m_vectors.size();

  if (m_textureType == TextureType::ImageTexture) {
    loadImageToDeviceTexture(m_h_genericTextureDevice.m_textureObject, status);
    if (status != StatusCode::NoError) {
      return;
    }
  }
  else if (m_textureType == TextureType::SolidColor) {
    // vectors
    int dataSize = m_vectors.size()*sizeof(float3);
    status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericTextureDevice.m_vectors), dataSize));
    if (status != StatusCode::NoError) {
      return;
    }
    status = CCE(cudaMemcpy(m_h_genericTextureDevice.m_vectors, m_vectors.data(),
			    dataSize, cudaMemcpyHostToDevice));
    if (status != StatusCode::NoError) {
      return;
    }
  }

  // whole texture
  status = CCE(cudaMemcpy(d_genericTextureDevice, &m_h_genericTextureDevice,
			  sizeof(GenericTextureDevice), cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }
}


GenericTexture::GenericTexture(const pt::ptree& texture,
			       StatusCode& status)
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
				     StatusCode& status)
{
  float3 albedo = string2float3(texture.get<std::string>("albedo.<xmlattr>.value"));
  m_vectors = {albedo};
}


void GenericTexture::parseImageTexture(const pt::ptree& texture,
				       StatusCode& status)
{
  status = StatusCode::NoError;
  TextureImageLoader textureImageLoader(texture);
  textureImageLoader.loadImage(m_imageWidth, m_imageHeight, m_numChannels,
  			       m_imageBuffer, status);
}
