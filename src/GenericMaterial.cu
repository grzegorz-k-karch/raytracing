#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericMaterial.h"
#include "GenericTexture.h"

void GenericMaterial::copyToDevice(GenericMaterialDevice* genericMaterialDevice,
				   StatusCodes& status) const
{
  status = StatusCodes::NoError;

  GenericMaterialDevice h_genericMaterialDevice;

  h_genericMaterialDevice.m_materialType = m_materialType;
  h_genericMaterialDevice.m_numScalars = m_scalars.size();
  h_genericMaterialDevice.m_numVectors = m_vectors.size();
  h_genericMaterialDevice.m_numTextures = m_textures.size();

  
  LOG_TRIVIAL(trace) << "GenericMaterial::copyToDevice: m_textures.size() = "
		     << m_textures.size();
  
  //--------------------------------------------------------------------------
  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.m_scalars),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterialDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
  //--------------------------------------------------------------------------
  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.m_vectors),
			  dataSize));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(h_genericMaterialDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  //--------------------------------------------------------------------------
  // textures
  dataSize = m_textures.size()*sizeof(GenericTextureDevice);
  status = CCE(cudaMalloc((void**)&(h_genericMaterialDevice.m_textures),
			  dataSize));
  for (int texIdx = 0; texIdx < m_textures.size(); texIdx++) {
    m_textures[texIdx].copyToDevice(h_genericMaterialDevice.m_textures + texIdx,
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

