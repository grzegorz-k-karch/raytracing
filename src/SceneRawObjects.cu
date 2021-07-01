#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneRawObjects.h"

SceneRawObjectsDevice* SceneRawObjects::copyToDevice(StatusCodes& status) const
{
  status = StatusCodes::NoError;

  // allocate pointer to sceneObjectsDevice on host -> h_sceneObjectsDevice
  SceneRawObjectsDevice h_sceneRawObjectsDevice;

  // copy objects
  int numObjects = m_objects.size();
  h_sceneRawObjectsDevice.m_numObjects = numObjects;

  //   allocate buffer for GenericObjectDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.m_objects),
			  numObjects*sizeof(GenericObjectDevice)));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }

  // go over all generic objects and copy them to the allocated
  // buffer h_sceneRawObjectsDevice.objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject =
      &(h_sceneRawObjectsDevice.m_objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject, status);
    if (status != StatusCodes::NoError) {
      return nullptr;
    }
  }

  // copy material to device
  int numMaterials = m_materials.size();
  h_sceneRawObjectsDevice.m_numMaterials = numMaterials;

  //   allocate buffer for GenericMaterialDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.m_materials),
			  numMaterials*sizeof(GenericMaterialDevice)));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }

  // go over all generic materials and copy them to the allocated
  // buffer h_sceneRawObjectsDevice.materials
  for (int objIdx = 0; objIdx < numMaterials; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying material " << objIdx;
    GenericMaterialDevice *currGenericMaterial =
      &(h_sceneRawObjectsDevice.m_materials[objIdx]);
    m_materials[objIdx].copyToDevice(currGenericMaterial, status);
    if (status != StatusCodes::NoError) {
      return nullptr;
    }
  }

  // copy camera to device
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.m_camera),
			  sizeof(Camera)));
  m_camera.copyToDevice(h_sceneRawObjectsDevice.m_camera, status);
  if (status != StatusCodes::NoError) {
    return nullptr;
  }

  // allocate pointer to sceneRawObjectsDevice on device
  SceneRawObjectsDevice *d_sceneRawObjectsDevice;
  status = CCE(cudaMalloc((void**)&d_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice)));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }
  status = CCE(cudaMemcpy(d_sceneRawObjectsDevice, &h_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice),
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }
  return d_sceneRawObjectsDevice;
}
