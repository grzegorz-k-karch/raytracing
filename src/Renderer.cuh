#ifndef RENDERER_CUH
#define RENDERER_CUH

#include "logging.h"
#include <curand_kernel.h>
#include <vector>
#include "SceneDevice.cuh"
#include "cuda_utils.cuh"

class Renderer {
public:
  Renderer(int imageWidth, int imageHeight, int sampleCount, int rayDepth) :
    m_imageWidth(imageWidth),
    m_imageHeight(imageHeight),
    m_sampleCount(sampleCount),
    m_rayDepth(rayDepth),
    m_framebuffer(nullptr),
    m_randState(nullptr) {}

  ~Renderer() {
    LOG_TRIVIAL(trace) << "Renderer::~Renderer";
    if (m_framebuffer) {
      LOG_TRIVIAL(trace) << "\treleasing m_framebuffer";
      CCE(cudaFree(m_framebuffer));
      m_framebuffer = nullptr;
    }
    if (m_randState) {
      LOG_TRIVIAL(trace) << "\treleasing m_randState";      
      CCE(cudaFree(m_randState));
      m_randState = nullptr;
    }
  }

  void initBuffers(StatusCode &status);
  void renderScene(const SceneDevice& sceneDevice, StatusCode& status);
  void getImageOnHost(std::vector<float3>& image, StatusCode& status) const;

private:
  void initRandState(StatusCode &status);

  int m_imageWidth;
  int m_imageHeight;
  int m_sampleCount;
  int m_rayDepth;
  curandState *m_randState;
  float3 *m_framebuffer;
};

#endif//RENDERER_CUH
