#ifndef RENDERER_CUH
#define RENDERER_CUH

#include "logging.h"
#include <curand_kernel.h>
#include <vector>
#include "SceneDevice.cuh"

class Renderer {
public:
  Renderer(int imageWidth, int imageHeight, int sampleCount)
      : m_imageWidth(imageWidth),
	m_imageHeight(imageHeight),
        m_sampleCount(sampleCount),
	m_framebuffer(nullptr),
	m_randState(nullptr) {
    LOG_TRIVIAL(trace) << "Renderer::Renderer";
  }

  ~Renderer() {
    LOG_TRIVIAL(trace) << "Renderer::~Renderer";
    if (m_framebuffer) {
      LOG_TRIVIAL(trace) << "\treleasing m_framebuffer";
      cudaFree(m_framebuffer);
      m_framebuffer = nullptr;
    }
    if (m_randState) {
      LOG_TRIVIAL(trace) << "\treleasing m_randState";      
      cudaFree(m_randState);
      m_randState = nullptr;
    }
  }

  void initBuffers(StatusCodes &status);

  void renderScene(const SceneDevice& sceneDevice, StatusCodes& status);

  void getImageOnHost(std::vector<float3>& image, StatusCodes& status) const;

private:
  void initRandState(StatusCodes &status);

  int m_imageWidth;
  int m_imageHeight;
  int m_sampleCount;
  curandState *m_randState;
  float3 *m_framebuffer;
};

#endif//RENDERER_CUH
