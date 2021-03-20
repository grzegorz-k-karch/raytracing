#include "GenericMaterial.h"

void GenericMaterial::copyToDevice(GenericMaterialDevice* genericMaterialDevice)
{
  GenericMaterialDevice *h_genericMaterial = new GenericMaterialDevice;

  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  CCE(cudaMalloc((void**)&(h_genericMaterial->scalars), dataSize));
  CCE(cudaMemcpy(h_genericMaterial->scalars, m_scalars.data(),
	     dataSize, cudaMemcpyHostToDevice);
  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  CCE(cudaMalloc((void**)&(h_genericMaterial->vectors), dataSize));
  CCE(cudaMemcpy(h_genericMaterial->vectors, m_vectors.data(),
	     dataSize, cudaMemcpyHostToDevice);

  h_genericMaterial->numScalars = m_scalars.size();
  h_genericMaterial->numVectors = m_vectors.size();
  
  // whole material
  CCE(cudaMemcpy(genericMaterialDevice, h_genericMaterial,
	     sizeof(GenericMaterialDevice), cudaMemcpyHostToDevice);
  
  delete h_genericMaterial;
}

