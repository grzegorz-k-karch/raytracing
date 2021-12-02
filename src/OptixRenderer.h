#ifndef OPTIX_RENDERER_H
#define OPTIX_RENDERER_H

#include <vector>
#include <optix.h>
#include "Camera.cuh"
#include "StatusCode.h"
#include "OptixRenderer.cuh"

template <typename T>
struct SBTRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SBTRecord<RayGenData>     RayGenSBTRecord;
typedef SBTRecord<MissData>       MissSBTRecord;
typedef SBTRecord<HitGroupData>   HitGroupSBTRecord;


class OptixRenderer {
 public:

  OptixRenderer(StatusCode& status);
  ~OptixRenderer();

  StatusCode createContext();
  StatusCode createModule(OptixPipelineCompileOptions& pipelineCompileOptions);
  StatusCode createProgramGroups();
  StatusCode createPipeline(OptixPipelineCompileOptions& pipelineCompileOptions);
  StatusCode setupShaderBindingTable(std::vector<HitGroupSBTRecord>& hitgroupRecords);
  StatusCode launch(const Camera& camera, std::vector<float3>& outputBuffer,
	      int imageWidth, int imageHeight);
  StatusCode buildRootAccelStruct(OptixTraversableHandle& traversableHandle);
  OptixDeviceContext getContext() const {
    return m_context;
  }

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
  CUdeviceptr m_d_gasOutputBuffer;  
};

#endif //OPTIX_RENDERER_H
