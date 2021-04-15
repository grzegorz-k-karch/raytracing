#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"

void GenericObject::copyToDevice(GenericObjectDevice* genericObjectDevice,
				 StatusCodes& status)
{
  status = StatusCodes::NoError;

  GenericObjectDevice h_genericObject;

  // copy all sizes and object type ------------------------------------------
  h_genericObject.numScalars = m_scalars.size();
  h_genericObject.numVectors = m_vectors.size();
  h_genericObject.numVertices = m_vertices.size();
  h_genericObject.numVertexColors = m_vertexColors.size();
  h_genericObject.numVertexNormals = m_vertexNormals.size();
  h_genericObject.numTriangleIndices = m_triangleIndices.size();
  h_genericObject.objectType = m_objectType;

  // scalars -----------------------------------------------------------------
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericObject.scalars), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObject.vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObject.vertices), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.vertices, m_vertices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex colors -----------------------------------------------------------
  dataSize = m_vertexColors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObject.vertexColors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.vertexColors, m_vertexColors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObject.vertexNormals), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.vertexNormals, m_vertexNormals.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_triangleIndices.size()*sizeof(int);
  status = CCE(cudaMalloc((void**)&(h_genericObject.triangleIndices), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObject.triangleIndices, m_triangleIndices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // whole object ------------------------------------------------------------
  status = CCE(cudaMemcpy(genericObjectDevice, &h_genericObject,
			  sizeof(GenericObjectDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}
