#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"
#include "GenericMaterial.h"

void GenericObject::copyToDevice(GenericObjectDevice* genericObjectDevice,
				 StatusCodes& status)
{
  status = StatusCodes::NoError;

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
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_material),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  m_material->copyToDevice(m_h_genericObjectDevice.m_material, status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // scalars -----------------------------------------------------------------
  dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_scalars), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_vertices),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertices, m_vertices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex colors -----------------------------------------------------------
  dataSize = m_vertexColors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_vertexColors),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertexColors,
			  m_vertexColors.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_vertexNormals),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_vertexNormals,
			  m_vertexNormals.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // texture coords ----------------------------------------------------------
  dataSize = m_textureCoords.size()*sizeof(float2);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_textureCoords),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_textureCoords,
			  m_textureCoords.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_triangleIndices.size()*sizeof(int);
  status = CCE(cudaMalloc((void**)&(m_h_genericObjectDevice.m_triangleIndices),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericObjectDevice.m_triangleIndices,
			  m_triangleIndices.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // whole object ------------------------------------------------------------
  status = CCE(cudaMemcpy(genericObjectDevice, &m_h_genericObjectDevice,
			  sizeof(GenericObjectDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
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


GenericObjectDevice::GenericObjectDevice(GenericObjectDevice&& other):
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
