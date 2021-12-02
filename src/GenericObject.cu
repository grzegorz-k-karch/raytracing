#include <optix.h>
#include <optix_stubs.h>

#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"

StatusCode GenericObject::copyAttributesToDevice(GenericObjectDevice& h_genericObjectDevice)
{
  StatusCode status = StatusCode::NoError;

  // copy all sizes and object type ------------------------------------------
  h_genericObjectDevice.m_bmin = m_bbox.min();
  h_genericObjectDevice.m_bmax = m_bbox.max();
  h_genericObjectDevice.m_numScalars = m_scalars.size();
  h_genericObjectDevice.m_numVectors = m_vectors.size();
  h_genericObjectDevice.m_numVertices = m_vertices.size();
  h_genericObjectDevice.m_albedo = m_albedo;
  h_genericObjectDevice.m_numVertexNormals = m_vertexNormals.size();
  h_genericObjectDevice.m_numTextureCoords = m_textureCoords.size();
  h_genericObjectDevice.m_numIndexTriplets = m_indexTriplets.size();
  h_genericObjectDevice.m_objectType = m_objectType;

  // scalars -----------------------------------------------------------------
  size_t dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_scalars), dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_vectors), dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_vertices),
			  dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(reinterpret_cast<void*>(h_genericObjectDevice.m_vertices),
			  m_vertices.data(),
			  dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_vertexNormals),
			  dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vertexNormals,
			  m_vertexNormals.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }

  // texture coords ----------------------------------------------------------
  dataSize = m_textureCoords.size()*sizeof(float2);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_textureCoords),
			  dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_textureCoords,
			  m_textureCoords.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_indexTriplets.size()*sizeof(uint3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&h_genericObjectDevice.m_indexTriplets),
			  dataSize));
  if (status != StatusCode::NoError) {
    return status;
  }
  status = CCE(cudaMemcpy(reinterpret_cast<void*>(h_genericObjectDevice.m_indexTriplets),
			  m_indexTriplets.data(),
			  dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return status;
  }
  return status;
}


GenericObjectDevice::~GenericObjectDevice()
{
  LOG_TRIVIAL(trace) << "~GenericObjectDevice";
  m_objectType = ObjectType::None;
  m_bmin = make_float3(0.0f, 0.0f, 0.0f);
  m_bmax = make_float3(0.0f, 0.0f, 0.0f);
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
    CCE(cudaFree(reinterpret_cast<void*>(m_vertices)));
    m_vertices = 0;
  }
  m_numVertices = 0;
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
    CCE(cudaFree(reinterpret_cast<void*>(m_indexTriplets)));
    m_indexTriplets = 0;
  }
  m_numIndexTriplets = 0;
}


GenericObjectDevice::GenericObjectDevice(GenericObjectDevice&& other) noexcept:
  m_objectType(other.m_objectType),
  m_bmin(other.m_bmin),
  m_bmax(other.m_bmax),
  m_scalars(other.m_scalars),
  m_numScalars(other.m_numScalars),
  m_vectors(other.m_vectors),
  m_numVectors(other.m_numVectors),
  m_vertices(other.m_vertices),
  m_numVertices(other.m_numVertices),
  m_albedo(other.m_albedo),
  m_vertexNormals(other.m_vertexNormals),
  m_numVertexNormals(other.m_numVertexNormals),
  m_textureCoords(other.m_textureCoords),
  m_numTextureCoords(other.m_numTextureCoords),
  m_indexTriplets(other.m_indexTriplets),
  m_numIndexTriplets(other.m_numIndexTriplets)
{
  other.m_scalars = nullptr;
  other.m_numScalars = 0;
  other.m_vectors = nullptr;
  other.m_numVectors = 0;
  other.m_vertices = 0;
  other.m_numVertices = 0;
  other.m_vertexNormals = nullptr;
  other.m_numVertexNormals = 0;
  other.m_textureCoords = nullptr;
  other.m_numTextureCoords = 0;
  other.m_indexTriplets = 0;
  other.m_numIndexTriplets = 0;
}
