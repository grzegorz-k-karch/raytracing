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
    LOG_TRIVIAL(error) << "cudaMalloc error\n";
  }
  CUdeviceptr d_tempOuputBuffer;
  status = CCE(cudaMalloc(
			  reinterpret_cast<void**>(&d_tempOuputBuffer),
			  gasBufferSizes.outputSizeInBytes
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "cudaMalloc error\n";
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

  m_d_outputBuffers.push_back(d_tempOuputBuffer);
}


void SceneRawObjects::generateOptixBuildInput(GenericObjectDevice& genObjDev,
					      OptixBuildInput& buildInput)
{
  StatusCode status = StatusCode::NoError;

  if (genObjDev.m_objectType == ObjectType::Mesh) {

    buildInput.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.numVertices      = genObjDev.m_numVertices;
    buildInput.triangleArray.vertexBuffers    = &(genObjDev.m_vertices);
    buildInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    buildInput.triangleArray.numIndexTriplets = genObjDev.m_numIndexTriplets;
    buildInput.triangleArray.indexBuffer      = genObjDev.m_indexTriplets;
    buildInput.triangleArray.flags            = m_inputFlags;
    buildInput.triangleArray.numSbtRecords    = 1;
  }
}


void SceneRawObjects::generateTraversableHandles(OptixDeviceContext context,
						 std::vector<OptixTraversableHandle>& traversableHandles)
{
  for (GenericObjectDevice& genObjDev : m_objectsDevice) {
    OptixBuildInput buildInput = {};
    memset(&buildInput, 0, sizeof(OptixBuildInput));
    generateOptixBuildInput(genObjDev, buildInput);
    OptixTraversableHandle traversableHandle;
    buildAccelStruct(context, &buildInput, &traversableHandle);
    traversableHandles.push_back(traversableHandle);
  }
}
