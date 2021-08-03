#include <assert.h>

#include "logging.h"
#include "cuda_utils.cuh"
#include "Objects.cuh"
#include "SceneDevice.cuh"

__global__ void destroyWorld_kernel(Object **world)
{
  delete *world;
  *world = nullptr;
}

SceneDevice::~SceneDevice()
{
  LOG_TRIVIAL(trace) << "SceneDevice::~SceneDevice";
  if (m_camera) {
    LOG_TRIVIAL(trace) << "\treleasing m_camera";
    CCE(cudaFree(m_camera));
    m_camera = nullptr;
  }
  if (m_world) {
    LOG_TRIVIAL(trace) << "\treleasing m_world";
    destroyWorld_kernel<<<1,1>>>(m_world);
    CCE(cudaFree(m_world));
    m_world = nullptr;
  }
}


__global__ void constructScene_kernel(
    const SceneRawObjectsDevice *sceneRawObjectsDevice,
    Camera* camera,
    Object** world)
{
  *camera = *(sceneRawObjectsDevice->m_camera);

  int numObjects = sceneRawObjectsDevice->m_numObjects;
  Object **objects = new Object*[numObjects];

  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    GenericObjectDevice *genObjDev =
      &(sceneRawObjectsDevice->m_objects[objIdx]);
    objects[objIdx] = ObjectFactory::createObject(genObjDev);
  }

  *world = createBVH(objects, numObjects);
}

void SceneDevice::constructScene(SceneRawObjects& sceneRawObjects,
				 StatusCode& status)
{
  status = StatusCode::NoError;

  // allocate pointer to sceneRawObjectsDevice on device
  SceneRawObjectsDevice *d_sceneRawObjectsDevice;
  status = CCE(cudaMalloc((void**)&d_sceneRawObjectsDevice,
			  sizeof(SceneRawObjectsDevice)));
  if (status != StatusCode::NoError) {
    return;
  }
  sceneRawObjects.copyToDevice(d_sceneRawObjectsDevice, status);
  if (status != StatusCode::NoError) {
    return;
  }

  // construct the scene on device
  status = CCE(cudaMalloc((void**)&m_world, sizeof(Object*)));
  if (status != StatusCode::NoError) {
    return;
  }
  status = CCE(cudaMalloc((void**)&m_camera, sizeof(Camera)));
  if (status != StatusCode::NoError) {
    return;
  }
  constructScene_kernel<<<1,1>>>(d_sceneRawObjectsDevice, m_camera, m_world);
  status = CCE(cudaDeviceSynchronize());
  if (status != StatusCode::NoError) {
    return;
  }
}
