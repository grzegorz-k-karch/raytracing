#ifndef RENDERER_CUH
#define RENDERER_CUH

#include "SceneDevice.cuh"

class Renderer {
public:
  void renderScene(const SceneDevice& sceneDevice,
		   StatusCodes& status) {}
};

#endif//RENDERER_CUH
