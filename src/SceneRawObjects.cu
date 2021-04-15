#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneRawObjects.h"

void SceneRawObjects::copyToDevice(StatusCodes& status)
{
  status = StatusCodes::NoError;

  // allocate pointer to sceneObjectsDevice on host -> h_sceneObjectsDevice
  SceneRawObjectsDevice h_sceneRawObjectsDevice;

  // copy objects
  int numObjects = m_objects.size();
  h_sceneRawObjectsDevice.numObjects = numObjects;

  //   allocate buffer for GenericObjectDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.objects),
			  numObjects*sizeof(GenericObjectDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }

  // go over all generic objects and copy them to the allocated
  // buffer h_sceneRawObjectsDevice.objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject = &(h_sceneRawObjectsDevice.objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject, status);
    if (status != StatusCodes::NoError) {
      return;
    }
  }

  // copy material to device
  int numMaterials = m_materials.size();
  h_sceneRawObjectsDevice.numMaterials = numMaterials;

  //   allocate buffer for GenericMaterialDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.materials),
			  numMaterials*sizeof(GenericMaterialDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }

  // go over all generic materials and copy them to the allocated
  // buffer h_sceneRawObjectsDevice.materials
  for (int objIdx = 0; objIdx < numMaterials; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying material " << objIdx;
    GenericMaterialDevice *currGenericMaterial = &(h_sceneRawObjectsDevice.materials[objIdx]);
    m_materials[objIdx].copyToDevice(currGenericMaterial, status);
    if (status != StatusCodes::NoError) {
      return;
    }
  }

  // copy camera to device
  status = CCE(cudaMalloc((void**)&(h_sceneRawObjectsDevice.camera),
			  sizeof(Camera)));
  m_camera.copyToDevice(h_sceneRawObjectsDevice.camera, status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // allocate pointer to sceneRawObjectsDevice on device
  status = CCE(cudaMalloc((void**)&m_sceneRawObjectsDevice, sizeof(SceneRawObjectsDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(m_sceneRawObjectsDevice, &h_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }
}
