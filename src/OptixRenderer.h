#ifndef OPTIX_RENDERER_H
#define OPTIX_RENDERER_H

#include <vector>
#include <optix.h>
#include "StatusCode.h"

class OptixRenderer {
 public:

  OptixRenderer(StatusCode& status);
  ~OptixRenderer();

  void createContext(StatusCode& status);
  void createModule(OptixPipelineCompileOptions& pipelineCompileOptions,
		    StatusCode& status);
  void createProgramGroups(StatusCode& status);
  void createPipeline(OptixPipelineCompileOptions& pipelineCompileOptions,
		      StatusCode& status);
  void setupShaderBindingTable(StatusCode& status);
  void launch(std::vector<float3>& outputBuffer, StatusCode& status);
  void buildRootAccelStruct(std::vector<OptixTraversableHandle>& traversableHandles,
			    StatusCode& status);
 private:
  OptixDeviceContext m_context;
  OptixModule m_module;
  OptixProgramGroup m_hitgroupProgramGroup;
  OptixProgramGroup m_missProgramGroup;
  OptixProgramGroup m_raygenProgramGroup;
  OptixPipeline m_pipeline;
  OptixShaderBindingTable m_shaderBindingTable;
  OptixTraversableHandle m_iasHandle;
  CUdeviceptr m_d_iasOutputBuffer;
};

#endif //OPTIX_RENDERER_H
