#include <optix.h>
// #include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"
#include "GenericMaterial.h"

void GenericObject::buildOptixAccelStruct(OptixDeviceContext context)
{
  // OptixTraversableHandle gasHandle;
  // CUdeviceptr            d_gasOutputBuffer;
  // {
  //   // Use default options for simplicity.  In a real use case we would want to
  //   // enable compaction, etc
  //   OptixAccelBuildOptions accelOptions = {};
  //   accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
  //   accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;


  //   const size_t verticesSize = sizeof(float3)*m_vertices.size();
  //   CUdeviceptr d_vertices=0;
  //   CCE(cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSize));
  //   CCE(cudaMemcpy(reinterpret_cast<void*>(d_vertices), m_vertices.data(),
  // 		   verticesSize, cudaMemcpyHostToDevice));

  //   // Our build input is a simple list of non-indexed triangle vertices
  //   const uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
  //   OptixBuildInput triangleInput = {};
  //   triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  //   triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
  //   triangleInput.triangleArray.numVertices   = static_cast<uint32_t>(m_vertices.size());
  //   triangleInput.triangleArray.vertexBuffers = &d_vertices;
  //   triangleInput.triangleArray.indexBuffer   = &d_triangleIndices;
  //   triangleInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_triangleIndices.size()/3);
  //   triangleInput.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3; 
  //   triangleInput.triangleArray.flags         = triangleInputFlags;
  //   triangleInput.triangleArray.numSbtRecords = 1;
    

  //   OptixAccelBufferSizes gas_buffer_sizes;
  //   // TODO: optix check
  //   optixAccelComputeMemoryUsage(
  // 				 context,
  // 				 &accelOptions,
  // 				 &triangleInput,
  // 				 1, // Number of build inputs
  // 				 &gas_buffer_sizes
  // 				 );
  //   CUdeviceptr d_temp_buffer_gas;
  //   CCE(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
  // 		   gas_buffer_sizes.tempSizeInBytes));
  //   CCE(cudaMalloc(reinterpret_cast<void**>(&d_gasOutputBuffer),
  // 		   gas_buffer_sizes.outputSizeInBytes));

  //   // TODO: optix check
  //   optixAccelBuild(
  // 		    context,
  // 		    0,                  // CUDA stream
  // 		    &accelOptions,
  // 		    &triangleInput,
  // 		    1,                  // num build inputs
  // 		    d_temp_buffer_gas,
  // 		    gas_buffer_sizes.tempSizeInBytes,
  // 		    d_gasOutputBuffer,
  // 		    gas_buffer_sizes.outputSizeInBytes,
  // 		    &gasHandle,
  // 		    nullptr,            // emitted property list
  // 		    0                   // num emitted properties
  // 		    );

  //   CCE(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
  //   CCE(cudaFree(reinterpret_cast<void*>(d_vertices)));
  // }
  
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
  m_h_genericObjectDevice.m_numTriangleIndices = m_triangleIndices.size();
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
  dataSize = m_triangleIndices.size()*sizeof(int);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericObjectDevice.m_triangleIndices),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_triangleIndices,
			  m_triangleIndices.data(), dataSize,
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
  if (m_triangleIndices) {
    CCE(cudaFree(m_triangleIndices));
    m_triangleIndices = nullptr;
  }
  m_numTriangleIndices = 0;
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
  m_triangleIndices(other.m_triangleIndices),
  m_numTriangleIndices(other.m_numTriangleIndices)
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
  other.m_triangleIndices = nullptr;
  other.m_numTriangleIndices = 0;
}
