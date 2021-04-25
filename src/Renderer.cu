#include <curand_kernel.h>

#include "cuda_utils.cuh"
#include "vector_utils.h"
#include "Renderer.cuh"


__global__
void initRandState_kernel(int imageWidth, int imageHeight, curandState* randState)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < imageWidth && j < imageHeight) {
    int pixelIdx = i + j*imageWidth;
    curand_init(1984, pixelIdx, 0, &randState[pixelIdx]);
  }
}

void Renderer::initRandState(StatusCodes &status)
{
  const int tx{8};
  const int ty{8};

  dim3 numThreads(tx, ty);
  dim3 numBlocks((m_imageWidth + tx - 1)/tx,
		 (m_imageHeight + ty - 1)/ty);

  status = CCE(cudaMalloc((void**)&m_randState, m_imageWidth*m_imageHeight*sizeof(curandState)));
  if (status != StatusCodes::NoError) {
    return;
  }
  initRandState_kernel<<<numBlocks, numThreads>>>(m_imageWidth, m_imageHeight, m_randState);
  status = CCE(cudaGetLastError());
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaDeviceSynchronize());
  if (status != StatusCodes::NoError) {
    return;
  }
}

void Renderer::initBuffers(StatusCodes &status)
{
  // initialize random state on device
  initRandState(status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // allocate  buffer for the final image
  int framebufferSize = m_imageWidth*m_imageHeight*sizeof(float3);
  status = CCE(cudaMallocManaged((void**)&m_framebuffer, framebufferSize));
  if (status != StatusCodes::NoError) {
    return;
  }
}

__global__
void renderScene_kernel(Camera* camera, Object** world,
			curandState* randState, int imageWidth,
			int imageHeight, int sampleCount,
			float3* framebuffer)
{
  int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
  int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
  
  if (pixelX < imageWidth && pixelY < imageHeight) {

    int pixelIdx = pixelX + pixelY*imageWidth;
    curandState localRandState = randState[pixelIdx];
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int sample = 0; sample < sampleCount; sample++) {
      float u = float(pixelX +
		      curand_uniform(&localRandState))/float(imageWidth);
      float v = float(pixelY +
		      curand_uniform(&localRandState))/float(imageHeight);
      Ray ray = camera->getRay(u, v, &localRandState);
      // color += getColor(ray, *world, &localRandState);
    }
    framebuffer[pixelIdx] = color/float(sampleCount);
  }
}

void Renderer::renderScene(const SceneDevice &sceneDevice,
                           StatusCodes &status)
{
  const int tx = 8;
  const int ty = 8;

  dim3 numThreads(tx, ty);
  dim3 numBlocks((m_imageWidth + tx - 1)/tx,
		 (m_imageHeight + ty - 1)/ty);

  renderScene_kernel<<<numBlocks, numThreads>>>(sceneDevice.m_camera,
						sceneDevice.m_world,
						m_randState, m_imageWidth,
						m_imageHeight, m_sampleCount,
						m_framebuffer);
  status = CCE(cudaDeviceSynchronize());
  if (status != StatusCodes::NoError) {
    return;
  }
}

void Renderer::getImageOnHost(std::vector<float3>& image, StatusCodes& status) const
{
  int imageSize = m_imageWidth*m_imageHeight;
  image.resize(imageSize);

  for (int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++) {
    image[pixelIdx] = m_framebuffer[pixelIdx];
  }
}
