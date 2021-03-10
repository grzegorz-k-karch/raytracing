#include "device_utils.cuh"
#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"

void copyToDevice(GenericObjectDevice* genericObjectDevice)
{
  GenericObjectDevice *h_genericObject = new GenericObjectDevice;

  int dataSize = m_scalars.size()*sizeof(float);
  cudaMalloc((void**)&(h_genericObjects->scalars), dataSize);
  cudaMemcpy(h_genericObjects->scalars, m_scalars.data(),
	     dataSize, cudaMemcpyHostToDevice);
  
  dataSize = m_vectors.size()*sizeof(float3);
  cudaMalloc((void**)&(h_genericObjects->vectors), dataSize);
  cudaMemcpy(h_genericObjects->vectors, m_vectors.data(),
	     dataSize, cudaMemcpyHostToDevice);

  dataSize = m_vertices.size()*sizeof(float3);
  cudaMalloc((void**)&(h_genericObjects->vertices), dataSize);
  cudaMemcpy(h_genericObjects->vertices, m_vertices.data(),
	     dataSize, cudaMemcpyHostToDevice);

  dataSize = m_vertexColors.size()*sizeof(float3);
  cudaMalloc((void**)&(h_genericObjects->vertexColors), dataSize);
  cudaMemcpy(h_genericObjects->vertexColors, m_vertexColors.data(),
	     dataSize, cudaMemcpyHostToDevice);
  
  dataSize = m_vertexNormals.size()*sizeof(float3);
  cudaMalloc((void**)&(h_genericObjects->vertexNormals), dataSize);
  cudaMemcpy(h_genericObjects->vertexNormals, m_vertexNormals.data(),
	     dataSize, cudaMemcpyHostToDevice);

  dataSize = m_triangleIndices.size()*sizeof(int);
  cudaMalloc((void**)&(h_genericObjects->triangleIndices), dataSize);
  cudaMemcpy(h_genericObjects->triangleIndices, m_triangleIndices.data(),
	     dataSize, cudaMemcpyHostToDevice);

  GenericObjectDevice *d_genericObject;
  cudaMalloc((void**)&d_genericObject, sizeof(GenericObjectDevice));
  cudaMemcpy(d_genericObject, h_genericObject, sizeof(GenericObjectDevice), cudaMemcpyHostToDevice);
  
  delete h_genericObject;
}

