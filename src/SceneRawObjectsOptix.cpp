#include <unordered_set>

#include "logging.h"
#include "cuda_utils.cuh"
#include "optix_utils.cuh"
#include "SceneRawObjects.h"

void SceneRawObjects::buildAccelStruct(OptixDeviceContext context,
				       OptixBuildInput* buildInput,				       
				       OptixTraversableHandle* traversableHandle)
{
  StatusCode status = StatusCode::NoError; // FIXME
  
  OptixAccelBuildOptions accelOptions = {};
  memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));  
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gasBufferSizes;
  status = OCE(optixAccelComputeMemoryUsage(
					    context,
					    &accelOptions,
					    buildInput,
					    1,  // num_build_inputs
					    &gasBufferSizes
					    ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  CUdeviceptr d_tempBuffer;
  status = CCE(cudaMalloc(
			  reinterpret_cast<void**>(&d_tempBuffer),
			  gasBufferSizes.tempSizeInBytes
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  CUdeviceptr d_tempOuputBuffer;
  status = CCE(cudaMalloc(
			  reinterpret_cast<void**>(&d_tempOuputBuffer),
			  gasBufferSizes.outputSizeInBytes
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  
  status = OCE(optixAccelBuild(
			       context,
			       0,              // CUDA stream
			       &accelOptions,
			       buildInput,
			       1,              // num build inputs
			       d_tempBuffer,
			       gasBufferSizes.tempSizeInBytes,
			       d_tempOuputBuffer,
			       gasBufferSizes.outputSizeInBytes,
			       traversableHandle,
			       nullptr, //&emitProperty,  // emitted property list
			       0 //1               // num emitted properties
			       ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  status = CCE(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  m_d_gasOutputBuffer = d_tempOuputBuffer;
}

void SceneRawObjects::generateTraversableHandles(
						 OptixDeviceContext context,
						 std::vector<OptixTraversableHandle>& traversableHandles
						 )
{
  std::unordered_set<ObjectType> typesInScene;
  int objIdx = 0;
  for (int objIdx = 0; objIdx < m_objects.size(); objIdx++) {
    typesInScene.insert(m_objects[objIdx].getObjectType());

    OptixBuildInput buildInput = {};
    memset(&buildInput, 0, sizeof(OptixBuildInput));
    m_objects[objIdx].generateOptixBuildInput(m_objectsDevice[objIdx], buildInput);
    OptixTraversableHandle traversableHandle;
    buildAccelStruct(context, &buildInput, &traversableHandle);
    traversableHandles.push_back(traversableHandle);
  }
  LOG_TRIVIAL(info) << "num types in scene: " << typesInScene.size() << "\n";
}
