#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneRawObjects.h"

void SceneRawObjects::copyToDevice(SceneRawObjectsDevice** sceneObjectsDevice,
				   StatusCodes& status)
{
  status = StatusCodes::NoError;

  // allocate pointer to sceneObjectsDevice on host -> h_sceneObjectsDevice
  SceneRawObjectsDevice *h_sceneObjectsDevice = new SceneRawObjectsDevice;

  // copy objects
  int numObjects = m_objects.size();
  h_sceneObjectsDevice->numObjects = numObjects;

  //   allocate buffer for GenericObjectDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneObjectsDevice->objects),
			  numObjects*sizeof(GenericObjectDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }

  // go over all generic objects and copy them to the allocated
  // buffer h_sceneObjectsDevice->objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject = &(h_sceneObjectsDevice->objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject, status);
    if (status != StatusCodes::NoError) {
      return;
    }
  }

  // copy material to device
  int numMaterials = m_materials.size();
  h_sceneObjectsDevice->numMaterials = numMaterials;

  //   allocate buffer for GenericMaterialDevice struct
  status = CCE(cudaMalloc((void**)&(h_sceneObjectsDevice->materials),
			  numMaterials*sizeof(GenericMaterialDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }

  // go over all generic materials and copy them to the allocated
  // buffer h_sceneObjectsDevice->materials
  for (int objIdx = 0; objIdx < numMaterials; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying material " << objIdx;
    GenericMaterialDevice *currGenericMaterial = &(h_sceneObjectsDevice->materials[objIdx]);
    m_materials[objIdx].copyToDevice(currGenericMaterial, status);
    if (status != StatusCodes::NoError) {
      return;
    }
  }

  // copy camera to device
  status = CCE(cudaMalloc((void**)&(h_sceneObjectsDevice->camera),
			  sizeof(Camera)));
  m_camera.copyToDevice(h_sceneObjectsDevice->camera, status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // allocate pointer to sceneObjectsDevice on device
  status = CCE(cudaMalloc((void**)sceneObjectsDevice, sizeof(SceneRawObjectsDevice)));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMemcpy(*sceneObjectsDevice, h_sceneObjectsDevice,
			  sizeof(SceneRawObjectsDevice), cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return;
  }

  delete h_sceneObjectsDevice;
}
