#include "cuda_utils.cuh"
#include "GenericMaterial.h"

void GenericMaterial::copyToDevice(GenericMaterialDevice* genericMaterialDevice,
				   StatusCodes& status)
{
  status = StatusCodes::NoError;

  GenericMaterialDevice h_genericMaterial;

  h_genericMaterial.numScalars = m_scalars.size();
  h_genericMaterial.numVectors = m_vectors.size();
  h_genericMaterial.materialType = m_materialType;
  
  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericMaterial.scalars), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterial.scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericMaterial.vectors), dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterial.vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  // whole material
  status = CCE(cudaMemcpy(genericMaterialDevice, &h_genericMaterial,
			  sizeof(GenericMaterialDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}

