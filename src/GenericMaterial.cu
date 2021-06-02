#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericMaterial.h"
#include "GenericTexture.h"

void GenericMaterial::copyToDevice(GenericMaterialDevice* genericMaterialDevice,
				   StatusCodes& status) const
{
  status = StatusCodes::NoError;

  GenericMaterialDevice h_genericMaterialDevice;

  h_genericMaterialDevice.materialType = m_materialType;
  h_genericMaterialDevice.numScalars = m_scalars.size();
  h_genericMaterialDevice.numVectors = m_vectors.size();
  h_genericMaterialDevice.numTextures = m_textures.size();

  
  LOG_TRIVIAL(trace) << "GenericMaterial::copyToDevice: m_textures.size() = "
		     << m_textures.size();
  
  //--------------------------------------------------------------------------
  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.scalars),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterialDevice.scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
  //--------------------------------------------------------------------------
  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.vectors),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterialDevice.vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  //--------------------------------------------------------------------------
  // textures
  dataSize = m_textures.size()*sizeof(GenericTextureDevice);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.textures),
			  dataSize));
  for (int texIdx = 0; texIdx < m_textures.size(); texIdx++) {
    
    m_textures[texIdx].copyToDevice(h_genericMaterialDevice.textures + texIdx,
				    status);
  }

  //--------------------------------------------------------------------------
  // whole material
  status = CCE(cudaMemcpy(genericMaterialDevice, &h_genericMaterialDevice,
			  sizeof(GenericMaterialDevice),
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}

