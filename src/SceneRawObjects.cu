#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneRawObjects.h"

SceneRawObjectsDevice* SceneRawObjects::copyToDevice(StatusCodes& status)
{
  status = StatusCodes::NoError;

  // copy objects
  int numObjects = m_objects.size();
  m_h_sceneRawObjectsDevice.m_numObjects = numObjects;

  // allocate buffer for GenericObjectDevice struct
  status = CCE(cudaMalloc((void**)&(m_h_sceneRawObjectsDevice.m_objects),
			  numObjects*sizeof(GenericObjectDevice)));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }

  // go over all generic objects and copy them to the allocated
  // buffer m_h_sceneRawObjectsDevice.objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject =
      &(m_h_sceneRawObjectsDevice.m_objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject, status);
    if (status != StatusCodes::NoError) {
      return nullptr;
    }
  }

  // copy camera to device
  status = CCE(cudaMalloc((void**)&(m_h_sceneRawObjectsDevice.m_camera),
			  sizeof(Camera)));
  m_camera.copyToDevice(m_h_sceneRawObjectsDevice.m_camera, status);
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
  status = CCE(cudaMemcpy(d_sceneRawObjectsDevice, &m_h_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice),
			  cudaMemcpyHostToDevice));
  if (status != StatusCodes::NoError) {
    return nullptr;
  }
  return d_sceneRawObjectsDevice;
}
