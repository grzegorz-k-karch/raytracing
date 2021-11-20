#include <optix.h>
// #include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"
#include "GenericMaterial.h"

void GenericObject::generateOptixBuildInput(OptixBuildInput& buildInput)
{
  const uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

  memset(&buildInput, 0, sizeof(OptixBuildInput));

  LOG_TRIVIAL(trace) << "Generating Optix Build Input: ";
  if (m_objectType == ObjectType::Mesh) {
    LOG_TRIVIAL(trace) << "Mesh\n";

    // std::vector<float3> m_vertices2;
    // // std::vector<uint3> m_indexTriplets2;

    // m_vertices2.push_back(make_float3(-0.5f, -0.5f, 0.0f));
    // m_vertices2.push_back(make_float3(0.5f, -0.5f, 0.0f));
    // m_vertices2.push_back(make_float3(0.0f, 0.5f, 0.0f));
    // // m_indexTriplets2.push_back(make_uint3(0,1,2));

    // const size_t verticesSize = sizeof(float3)*m_vertices2.size();
    // CUdeviceptr d_vertices = 0;
    // CCE(cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSize));
    // CCE(cudaMemcpy(reinterpret_cast<void*>(d_vertices), m_vertices2.data(),
    // 		   verticesSize, cudaMemcpyHostToDevice));

    // // const size_t indexTripletsSize = sizeof(uint3)*m_indexTriplets2.size();
    // // CUdeviceptr d_indexTriplets = 0;
    // // CCE(cudaMalloc(reinterpret_cast<void**>(&d_indexTriplets), indexTripletsSize));
    // // CCE(cudaMemcpy(reinterpret_cast<void*>(d_indexTriplets), m_indexTriplets2.data(),
    // // 		   indexTripletsSize, cudaMemcpyHostToDevice));

    buildInput.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
    // buildInput.triangleArray.numVertices      = static_cast<uint32_t>(m_vertices2.size());
    // buildInput.triangleArray.vertexBuffers    = &d_vertices;
    // buildInput.triangleArray.indexBuffer      = d_indexTriplets;
    // buildInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_indexTriplets2.size());
    // buildInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.flags            = inputFlags;
    buildInput.triangleArray.numSbtRecords    = 1;
  }
  else if (m_objectType == ObjectType::Sphere) {
    LOG_TRIVIAL(trace) << "Sphere\n";
    // AABB build input
    OptixAabb aabb  = {m_bbox.min().x, m_bbox.min().y, m_bbox.min().z,
		       m_bbox.max().x, m_bbox.max().y, m_bbox.max().z};

    CUdeviceptr d_aabbBuffer;
    CCE(cudaMalloc(reinterpret_cast<void**>(&d_aabbBuffer), sizeof(OptixAabb)));
    CCE(cudaMemcpy(reinterpret_cast<void*>(d_aabbBuffer), &aabb, sizeof(OptixAabb),
			  cudaMemcpyHostToDevice));
    buildInput.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers   = &d_aabbBuffer;
    buildInput.customPrimitiveArray.numPrimitives = 1;
    buildInput.customPrimitiveArray.flags         = inputFlags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;
  }
}

void GenericObject::copyToDevice(GenericObjectDevice* d_genericObject,
				 StatusCode& status)
{
  status = StatusCode::NoError;

  // copy all sizes and object type ------------------------------------------
  m_h_genericObjectDevice.m_bmin = m_bbox.min();
  m_h_genericObjectDevice.m_bmax = m_bbox.max();
  m_h_genericObjectDevice.m_numScalars = m_scalars.size();
  m_h_genericObjectDevice.m_numVectors = m_vectors.size();
  m_h_genericObjectDevice.m_numVertices = m_vertices.size();
  m_h_genericObjectDevice.m_numVertexColors = m_vertexColors.size();
  m_h_genericObjectDevice.m_numVertexNormals = m_vertexNormals.size();
  m_h_genericObjectDevice.m_numTextureCoords = m_textureCoords.size();
  m_h_genericObjectDevice.m_numIndexTriplets = m_indexTriplets.size();
  m_h_genericObjectDevice.m_objectType = m_objectType;

  // material ----------------------------------------------------------------
  // allocate buffer for GenericMaterialDevice struct
  int dataSize = sizeof(GenericMaterialDevice);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_material),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  m_material->copyToDevice(m_h_genericObjectDevice.m_material, status);
  if (status != StatusCode::NoError) {
    return;
  }

  // scalars -----------------------------------------------------------------
  dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_scalars), dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_vectors), dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_vertices),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertices, m_vertices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // vertex colors -----------------------------------------------------------
  dataSize = m_vertexColors.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_vertexColors),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertexColors,
			  m_vertexColors.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_vertexNormals),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertexNormals,
			  m_vertexNormals.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // texture coords ----------------------------------------------------------
  dataSize = m_textureCoords.size()*sizeof(float2);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_textureCoords),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_textureCoords,
			  m_textureCoords.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_indexTriplets.size()*sizeof(uint3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_indexTriplets),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_indexTriplets,
			  m_indexTriplets.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  // whole object ------------------------------------------------------------
  status = CCE(cudaMemcpy(d_genericObject, &m_h_genericObjectDevice,
			  sizeof(GenericObjectDevice), cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }
}

GenericObjectDevice::~GenericObjectDevice()
{
  LOG_TRIVIAL(trace) << "~GenericObjectDevice";
  m_objectType = ObjectType::None;
  m_bmin = make_float3(0.0f, 0.0f, 0.0f);
  m_bmax = make_float3(0.0f, 0.0f, 0.0f);
  // m_material->releaseData();
  // m_material =  = nullptr;
  if (m_scalars) {
    CCE(cudaFree(m_scalars));
    m_scalars = nullptr;
  }
  m_numScalars = 0;
  if (m_vectors) {
    CCE(cudaFree(m_vectors));
    m_vectors = nullptr;
  }
  m_numVectors = 0;
  if (m_vertices) {
    CCE(cudaFree(m_vertices));
    m_vertices = nullptr;
  }
  m_numVertices = 0;
  if (m_vertexColors) {
    CCE(cudaFree(m_vertexColors));
    m_vertexColors = nullptr;
  }
  m_numVertexColors = 0;
  if (m_vertexNormals) {
    CCE(cudaFree(m_vertexNormals));
    m_vertexNormals = nullptr;
  }
  m_numVertexNormals = 0;
  if (m_textureCoords) {
    CCE(cudaFree(m_textureCoords));
    m_textureCoords = nullptr;
  }
  m_numTextureCoords = 0;
  if (m_indexTriplets) {
    CCE(cudaFree(m_indexTriplets));
    m_indexTriplets = nullptr;
  }
  m_numIndexTriplets = 0;
}


GenericObjectDevice::GenericObjectDevice(GenericObjectDevice&& other) noexcept:
  m_objectType(other.m_objectType),
  m_bmin(other.m_bmin),
  m_bmax(other.m_bmax),
  m_material(other.m_material),
  m_scalars(other.m_scalars),
  m_numScalars(other.m_numScalars),
  m_vectors(other.m_vectors),
  m_numVectors(other.m_numVectors),
  m_vertices(other.m_vertices),
  m_numVertices(other.m_numVertices),
  m_vertexColors(other.m_vertexColors),
  m_numVertexColors(other.m_numVertexColors),
  m_vertexNormals(other.m_vertexNormals),
  m_numVertexNormals(other.m_numVertexNormals),
  m_textureCoords(other.m_textureCoords),
  m_numTextureCoords(other.m_numTextureCoords),
  m_indexTriplets(other.m_indexTriplets),
  m_numIndexTriplets(other.m_numIndexTriplets)
{
  other.m_material = nullptr;
  other.m_scalars = nullptr;
  other.m_numScalars = 0;
  other.m_vectors = nullptr;
  other.m_numVectors = 0;
  other.m_vertices = nullptr;
  other.m_numVertices = 0;
  other.m_vertexColors = nullptr;
  other.m_numVertexColors = 0;
  other.m_vertexNormals = nullptr;
  other.m_numVertexNormals = 0;
  other.m_textureCoords = nullptr;
  other.m_numTextureCoords = 0;
  other.m_indexTriplets = nullptr;
  other.m_numIndexTriplets = 0;
}


// // Use default options for simplicity.  In a real use case we would want to
// // enable compaction, etc
// OptixAccelBuildOptions accelOptions = {};
// memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));
// accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
// accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
// accelOptions.motionOptions.numKeys = 0;



// OptixAccelBufferSizes gasBufferSizes;
// // TODO: optix check
// optixAccelComputeMemoryUsage(
// 			       context,
// 			       &accelOptions,
// 			       &triangleInput,
// 			       1, // Number of build inputs
// 			       &gasBufferSizes
// 			       );
// CUdeviceptr d_tempBufferGas;
// CCE(cudaMalloc(reinterpret_cast<void**>(&d_tempBufferGas),
// 		 gasBufferSizes.tempSizeInBytes));
// CCE(cudaMalloc(reinterpret_cast<void**>(&d_gasOutputBuffer),
// 		 gasBufferSizes.outputSizeInBytes));

// // TODO: optix check
// optixAccelBuild(
// 		  context,
// 		  0,                  // CUDA stream
// 		  &accelOptions,
// 		  &triangleInput,
// 		  1,                  // num build inputs
// 		  d_tempBufferGas,
// 		  gasBufferSizes.tempSizeInBytes,
// 		  d_gasOutputBuffer,
// 		  gasBufferSizes.outputSizeInBytes,
// 		  &gasHandle,
// 		  nullptr,            // emitted property list
// 		  0                   // num emitted properties
// 		  );

// CCE(cudaFree(reinterpret_cast<void*>(d_tempBufferGas)));
// CCE(cudaFree(reinterpret_cast<void*>(d_vertices)));
