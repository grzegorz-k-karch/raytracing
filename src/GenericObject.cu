#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericObject.h"

void GenericObject::copyToDevice(GenericObjectDevice* genericObjectDevice,
				 StatusCodes& status) const
{
  status = StatusCodes::NoError;

  GenericObjectDevice h_genericObjectDevice;

  // copy all sizes and object type ------------------------------------------
  h_genericObjectDevice.bmin = m_bbox.min();
  h_genericObjectDevice.bmax = m_bbox.max();  
  h_genericObjectDevice.numScalars = m_scalars.size();
  h_genericObjectDevice.numVectors = m_vectors.size();
  h_genericObjectDevice.numVertices = m_vertices.size();
  h_genericObjectDevice.numVertexColors = m_vertexColors.size();
  h_genericObjectDevice.numVertexNormals = m_vertexNormals.size();
  h_genericObjectDevice.numTriangleIndices = m_triangleIndices.size();
  h_genericObjectDevice.objectType = m_objectType;

  // scalars -----------------------------------------------------------------
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.scalars), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vectors -----------------------------------------------------------------
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertices ----------------------------------------------------------------
  dataSize = m_vertices.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.vertices), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.vertices, m_vertices.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex colors -----------------------------------------------------------
  dataSize = m_vertexColors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.vertexColors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.vertexColors, m_vertexColors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vertex normals ----------------------------------------------------------
  dataSize = m_vertexNormals.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.vertexNormals), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.vertexNormals, m_vertexNormals.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // triangle indices --------------------------------------------------------
  dataSize = m_triangleIndices.size()*sizeof(int);
  status = CCE(cudaMalloc((void**)&(h_genericObjectDevice.triangleIndices), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericObjectDevice.triangleIndices, m_triangleIndices.data(),
			  dataSize, cudaMemcpyHostToDevice));
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
