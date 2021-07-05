#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"
#include "GenericMaterial.h"

void GenericObject::copyToDevice(GenericObjectDevice* genericObjectDevice,
				 StatusCodes& status) const
{
  status = StatusCodes::NoError;

  GenericObjectDevice h_genericObjectDevice;

  // copy all sizes and object type ------------------------------------------
  h_genericObjectDevice.m_bmin = m_bbox.min();
  h_genericObjectDevice.m_bmax = m_bbox.max();
  h_genericObjectDevice.m_numScalars = m_scalars.size();
  h_genericObjectDevice.m_numVectors = m_vectors.size();
  h_genericObjectDevice.m_numVertices = m_vertices.size();
  h_genericObjectDevice.m_numVertexColors = m_vertexColors.size();
  h_genericObjectDevice.m_numVertexNormals = m_vertexNormals.size();
  h_genericObjectDevice.m_numTextureCoords = m_textureCoords.size();
  h_genericObjectDevice.m_numTriangleIndices = m_triangleIndices.size();
  h_genericObjectDevice.m_objectType = m_objectType;

  // material ----------------------------------------------------------------
  // allocate buffer for GenericMaterialDevice struct
  int dataSize = sizeof(GenericMaterialDevice);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_material),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  m_material->copyToDevice(h_genericObjectDevice.m_material, status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // scalars -----------------------------------------------------------------
  dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_scalars), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_vertices),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vertices, m_vertices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex colors -----------------------------------------------------------
  dataSize = m_vertexColors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_vertexColors),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vertexColors,
			  m_vertexColors.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_vertexNormals),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_vertexNormals,
			  m_vertexNormals.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // texture coords ----------------------------------------------------------
  dataSize = m_textureCoords.size()*sizeof(float2);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_textureCoords),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_textureCoords,
			  m_textureCoords.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_triangleIndices.size()*sizeof(int);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.m_triangleIndices),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.m_triangleIndices,
			  m_triangleIndices.data(), dataSize,
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // whole object ------------------------------------------------------------
  status = CCE(cudaMemcpy(genericObjectDevice, &h_genericObjectDevice,
			  sizeof(GenericObjectDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}
