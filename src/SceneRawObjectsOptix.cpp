#include <unordered_set>

#include "logging.h"
#include "cuda_utils.cuh"
#include "optix_utils.cuh"
#include "SceneRawObjects.h"

// template<class T>
// T roundUp(T a, T b)
// {
//   T c = (a + b - 1)/b;
//   return b*c;
// }

void SceneRawObjects::buildAccelStruct(OptixDeviceContext context,
				       OptixTraversableHandle* traversableHandle)
{
  StatusCode status = StatusCode::NoError; // FIXME
  
  OptixAccelBuildOptions accelOptions = {};
  memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));  
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

  const std::array<float3, 3> vertices =
    { {
       { -0.5f, -0.5f, 0.0f },
       {  0.5f, -0.5f, 0.0f },
       {  0.0f,  0.5f, 0.0f }
       } };

  const size_t verticesSize = sizeof(float3)*vertices.size();
  CUdeviceptr d_vertices = 0;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSize));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = CCE(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(),
			  verticesSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  OptixBuildInput buildInput = {};
  const uint32_t buildInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
  buildInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  buildInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
  buildInput.triangleArray.numVertices   = static_cast<uint32_t>(vertices.size());
  buildInput.triangleArray.vertexBuffers = &d_vertices;
  buildInput.triangleArray.flags         = buildInputFlags;
  buildInput.triangleArray.numSbtRecords = 1;

  OptixAccelBufferSizes gasBufferSizes;
  status = OCE(optixAccelComputeMemoryUsage(
					    context,
					    &accelOptions,
					    &buildInput,
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
			       &buildInput,
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
  status = CCE(cudaFree(reinterpret_cast<void*>(d_vertices)));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  m_d_triangleGasOutputBuffer = d_tempOuputBuffer;
}

void SceneRawObjects::generateTraversableHandles(OptixDeviceContext context,
						 std::vector<OptixTraversableHandle>& traversableHandles)
{
  std::unordered_set<ObjectType> typesInScene;
  for (auto& object : m_objects) {
    // OptixBuildInput buildInput;
    // object.generateOptixBuildInput(buildInput);
    typesInScene.insert(object.getObjectType());
    OptixTraversableHandle traversableHandle;
    buildAccelStruct(context, &traversableHandle);
    traversableHandles.push_back(traversableHandle);
  }
  LOG_TRIVIAL(info) << "num types in scene: " << typesInScene.size() << "\n";
}


#if 0
void SceneRawObjects::buildAccelStruct(OptixDeviceContext context,
				       OptixBuildInput buildInput,
				       OptixTraversableHandle& traversableHandle)
{
  std::vector<float3> m_vertices2;
  // std::vector<uint3> m_indexTriplets2;

  m_vertices2.push_back(make_float3(-0.5f, -0.5f, 0.0f));
  m_vertices2.push_back(make_float3(0.5f, -0.5f, 0.0f));
  m_vertices2.push_back(make_float3(0.0f, 0.5f, 0.0f));
  // m_indexTriplets2.push_back(make_uint3(0,1,2));

  const size_t verticesSize = sizeof(float3)*m_vertices2.size();
  CUdeviceptr d_vertices = 0;
  CCE(cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSize));
  CCE(cudaMemcpy(reinterpret_cast<void*>(d_vertices), m_vertices2.data(),
		 verticesSize, cudaMemcpyHostToDevice));
  buildInput.triangleArray.vertexBuffers    = &d_vertices;
  buildInput.triangleArray.numVertices      = static_cast<uint32_t>(m_vertices2.size());  
  
  StatusCode status = StatusCode::NoError;
  OptixAccelBuildOptions accelOptions = {};
  memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));  
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
  accelOptions.motionOptions.numKeys = 0;  

  OptixAccelBufferSizes gasBufferSizes;
  status = OCE(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput,
					    1,  // num_build_inputs
					    &gasBufferSizes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  CUdeviceptr d_tempBuffer;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  // non-compacted output
  CUdeviceptr d_tempOuputBuffer;
  status = CCE(cudaMalloc(			  reinterpret_cast<void**>(&d_tempOuputBuffer),
			  gasBufferSizes.outputSizeInBytes
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  
  // size_t      compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
  // CCE(cudaMalloc(
  // 		 reinterpret_cast<void**>(&d_bufferTempOutputGasAndCompactedSize),
  // 		 compactedSizeOffset + 8
  // 		 ));

  // OptixAccelEmitDesc emitProperty = {};
  // emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  // emitProperty.result = (CUdeviceptr)((char*)d_bufferTempOutputGasAndCompactedSize + compactedSizeOffset);


  status = OCE(optixAccelBuild(
			       context,
			       0,              // CUDA stream
			       &accelOptions,
			       &buildInput,
			       1,              // num build inputs
			       d_tempBuffer,
			       gasBufferSizes.tempSizeInBytes,
			       d_tempOuputBuffer,
			       gasBufferSizes.outputSizeInBytes,
			       &traversableHandle,
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

  // size_t compactedGasSize;
  // CCE(cudaMemcpy(
  // 		 &compactedGasSize,
  // 		 (void*)emitProperty.result,
  // 		 sizeof(size_t),
  // 		 cudaMemcpyDeviceToHost
  // 		 ));

  // if(compactedGasSize < gasBufferSizes.outputSizeInBytes) {
  //   CCE(cudaMalloc(reinterpret_cast<void**>(m_d_triangleGasOutputBuffer), compactedGasSize));

  //   // use handle as input and output
  //   LOG_TRIVIAL(trace) << "compactedGasSize = " << compactedGasSize;
  //   LOG_TRIVIAL(trace) << "gasBufferSizes.outputSizeInBytes = " << gasBufferSizes.outputSizeInBytes;
  //   OCE(optixAccelCompact(context, 0, traversableHandle, m_d_triangleGasOutputBuffer, compactedGasSize, &traversableHandle));

  //   CCE(cudaFree((void*)d_bufferTempOutputGasAndCompactedSize));
  // }
  // else {
  m_d_triangleGasOutputBuffer = d_tempOuputBuffer;
  // }
}

#endif
