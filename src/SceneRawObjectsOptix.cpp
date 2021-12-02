#include <unordered_set>

#include "logging.h"
#include "cuda_utils.cuh"
#include "optix_utils.cuh"
#include "SceneRawObjects.h"

void SceneRawObjects::buildAccelStruct(OptixDeviceContext context,
				       std::vector<OptixBuildInput>& buildInputs,
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
					    buildInputs.data(),
					    buildInputs.size(),  // num_build_inputs
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
			       buildInputs.data(),
			       buildInputs.size(), // num build inputs
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
    buildInput.triangleArray.sbtIndexOffsetBuffer        = 0; 
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0; 
  }
}


void SceneRawObjects::generateTraversableHandle(OptixDeviceContext context,
						OptixTraversableHandle* traversableHandle)
{
  std::vector<OptixBuildInput> buildInputs;
  buildInputs.resize(m_h_genericObjectsDevice.size());
      
  for (int objIdx = 0; objIdx < m_h_genericObjectsDevice.size(); objIdx++) {
    memset(&buildInputs[objIdx], 0, sizeof(OptixBuildInput));
    generateOptixBuildInput(m_h_genericObjectsDevice[objIdx], buildInputs[objIdx]);
  }
  buildAccelStruct(context, buildInputs, traversableHandle);
}


void SceneRawObjects::generateHitGroupRecords(std::vector<HitGroupSBTRecord>& hitgroupRecords)
{
  for (GenericObjectDevice& genObjDev : m_h_genericObjectsDevice) {
    HitGroupSBTRecord rec;
    rec.data.albedo = genObjDev.m_albedo;
    rec.data.normals = genObjDev.m_vertexNormals;
    rec.data.textureCoords = genObjDev.m_textureCoords;
    rec.data.indexTriplets = (uint3*)genObjDev.m_indexTriplets;
    hitgroupRecords.push_back(rec);
  }
}
