#include <assert.h>

#include "logging.h"
#include "cuda_utils.cuh"
#include "Objects.cuh"
#include "SceneDevice.cuh"


__global__ void constructScene_kernel(
    const SceneRawObjectsDevice *sceneRawObjectsDevice,
    Camera* camera,
    Object** world)
{
  *camera = *(sceneRawObjectsDevice->camera);

  int numObjects = sceneRawObjectsDevice->numObjects;
  Object **objects = new Object*[numObjects];

  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    GenericMaterialDevice *genMatDev =
      &(sceneRawObjectsDevice->materials[objIdx]);
    Material *material = MaterialFactory::createMaterial(genMatDev);
    GenericObjectDevice *genObjDev =
      &(sceneRawObjectsDevice->objects[objIdx]);
    objects[objIdx] = ObjectFactory::createObject(genObjDev, material);
  }

  *world = createBVH(objects, numObjects);
}

void SceneDevice::constructScene(const SceneRawObjects& sceneRawObjects,
				 StatusCodes& status)
{
  status = StatusCodes::NoError;

  SceneRawObjectsDevice *d_sceneRawObjectsDevice =
    sceneRawObjects.getObjectsOnDevice(status);
  if (status != StatusCodes::NoError) {
    return;
  }

  // construct the scene on device
  status = CCE(cudaMalloc((void**)&m_world, sizeof(Object*)));
  if (status != StatusCodes::NoError) {
    return;
  }
  status = CCE(cudaMalloc((void**)&m_camera, sizeof(Camera)));
  if (status != StatusCodes::NoError) {
    return;
  }
  constructScene_kernel<<<1,1>>>(d_sceneRawObjectsDevice, m_camera, m_world);
  status = CCE(cudaDeviceSynchronize());
  if (status != StatusCodes::NoError) {
    return;
  }
}
