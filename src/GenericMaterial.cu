#include "logging.h"
#include "cuda_utils.cuh"
#include "GenericMaterial.h"
#include "GenericTexture.h"

void GenericMaterial::copyToDevice(GenericMaterialDevice* d_genericMaterial,
				   StatusCode& status)
{
  LOG_TRIVIAL(trace) << "GenericMaterial::copyToDevice";
  
  status = StatusCode::NoError;

  m_h_genericMaterialDevice.m_materialType = m_materialType;
  m_h_genericMaterialDevice.m_numScalars = m_scalars.size();
  m_h_genericMaterialDevice.m_numVectors = m_vectors.size();
  m_h_genericMaterialDevice.m_numTextures = m_textures.size();
  
  //--------------------------------------------------------------------------
  // scalars
  int dataSize = m_scalars.size()*sizeof(float);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericMaterialDevice.m_scalars),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericMaterialDevice.m_scalars, m_scalars.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }
  //--------------------------------------------------------------------------
  // vectors
  dataSize = m_vectors.size()*sizeof(float3);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericMaterialDevice.m_vectors),
			  dataSize));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_h_genericMaterialDevice.m_vectors, m_vectors.data(),
			  dataSize, cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }

  //--------------------------------------------------------------------------
  // textures
  dataSize = m_textures.size()*sizeof(GenericTextureDevice);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_h_genericMaterialDevice.m_textures),
			  dataSize));
  for (int texIdx = 0; texIdx < m_textures.size(); texIdx++) {
    m_textures[texIdx].copyToDevice(m_h_genericMaterialDevice.m_textures + texIdx,
				    status);
  }

  //--------------------------------------------------------------------------
  // whole material
  status = CCE(cudaMemcpy(d_genericMaterial, &m_h_genericMaterialDevice,
			  sizeof(GenericMaterialDevice),
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }
}

GenericMaterialDevice::~GenericMaterialDevice()
{
  LOG_TRIVIAL(trace) << "~GenericMaterialDevice";
  m_materialType = MaterialType::None;
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
  if (m_textures) {
    CCE(cudaFree(m_textures));
    m_textures = nullptr;
  }
  m_numTextures = 0;
}
