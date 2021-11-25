#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneRawObjects.h"

void SceneRawObjects::copyToDevice(StatusCode& status)
{
  status = StatusCode::NoError;

  for (auto &object : m_objects) {
    GenericObjectDevice h_genericObjectDevice;
    object.copyAttributesToDevice(h_genericObjectDevice, status);
    m_objectsDevice.push_back(std::move(h_genericObjectDevice));
  }
}


void SceneRawObjects::copyToDevice(SceneRawObjectsDevice* d_sceneRawObjectsDevice,
				   StatusCode& status)
{
  status = StatusCode::NoError;

  // copy objects
  int numObjects = m_objects.size();
  m_h_sceneRawObjectsDevice.m_numObjects = numObjects;

  // allocate buffer for GenericObjectDevice struct
  status = CCE(cudaMalloc((void**)&(m_h_sceneRawObjectsDevice.m_objects),
			  numObjects*sizeof(GenericObjectDevice)));
  if (status != StatusCode::NoError) {
    return;
  }

  // go over all generic objects and copy them to the allocated
  // buffer m_h_sceneRawObjectsDevice.objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject =
      &(m_h_sceneRawObjectsDevice.m_objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject, status);
    if (status != StatusCode::NoError) {
      return;
    }
  }

  // copy camera to device
  status = CCE(cudaMalloc((void**)&(m_h_sceneRawObjectsDevice.m_camera),
			  sizeof(Camera)));
  m_camera.copyToDevice(m_h_sceneRawObjectsDevice.m_camera, status);
  if (status != StatusCode::NoError) {
    return;
  }

  // copy the whole scenerawobjects
  status = CCE(cudaMemcpy(d_sceneRawObjectsDevice, &m_h_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice),
			  cudaMemcpyHostToDevice));
  if (status != StatusCode::NoError) {
    return;
  }
}
