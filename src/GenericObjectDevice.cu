#include "GenericObject.h"

void GenericObject::copyToDevice(GenericObjectDevice* genericObjectDevice)
{
  GenericObjectDevice *h_genericObject = new GenericObjectDevice;

  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  CCE(cudaMalloc((void**)&(h_genericObject->scalars), dataSize));
  CCE(cudaMemcpy(h_genericObject->scalars, m_scalars.data(),
		 dataSize, cudaMemcpyHostToDevice));

  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  CCE(cudaMalloc((void**)&(h_genericObject->vectors), dataSize));
  CCE(cudaMemcpy(h_genericObject->vectors, m_vectors.data(),
		 dataSize, cudaMemcpyHostToDevice));

  // vertices
  dataSize = m_vertices.size()*sizeof(float3);
  CCE(cudaMalloc((void**)&(h_genericObject->vertices), dataSize));
  CCE(cudaMemcpy(h_genericObject->vertices, m_vertices.data(),
		 dataSize, cudaMemcpyHostToDevice));

  // vertex colors
  dataSize = m_vertexColors.size()*sizeof(float3);
  CCE(cudaMalloc((void**)&(h_genericObject->vertexColors), dataSize));
  CCE(cudaMemcpy(h_genericObject->vertexColors, m_vertexColors.data(),
		 dataSize, cudaMemcpyHostToDevice));

  // vertex normals
  dataSize = m_vertexNormals.size()*sizeof(float3);
  CCE(cudaMalloc((void**)&(h_genericObject->vertexNormals), dataSize));
  CCE(cudaMemcpy(h_genericObject->vertexNormals, m_vertexNormals.data(),
		 dataSize, cudaMemcpyHostToDevice));

  // triangle indices
  dataSize = m_triangleIndices.size()*sizeof(int);
  CCE(cudaMalloc((void**)&(h_genericObject->triangleIndices), dataSize));
  CCE(cudaMemcpy(h_genericObject->triangleIndices, m_triangleIndices.data(),
		 dataSize, cudaMemcpyHostToDevice));

  h_genericObject->numScalars = m_scalars.size();
  h_genericObject->numVectors = m_vectors.size();
  h_genericObject->numVertices = m_vertices.size();
  h_genericObject->numVertexColors = m_vertexColors.size();
  h_genericObject->numVertexNormals = m_vertexNormals.size();
  h_genericObject->numTriangleIndices = m_triangleIndices.size();

  // whole object
  CCE(cudaMemcpy(genericObjectDevice, h_genericObject,
		 sizeof(GenericObjectDevice), cudaMemcpyHostToDevice));

  delete h_genericObject;
}
