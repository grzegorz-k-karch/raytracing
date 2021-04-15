#include "logging.h"
#include "cuda_utils.cuh"
#include "SceneDevice.cuh"

#include <assert.h>

__global__
void constructScene_kernel(const SceneRawObjectsDevice *sceneRawObjectsDevice,
			   Object** world)
{
  int numObjects = sceneRawObjectsDevice->numObjects;
  Object **objectList = new Object*[numObjects];

  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    GenericMaterialDevice *genMatDev =
      &(sceneRawObjectsDevice->materials[objIdx]);
    Material *mat = MaterialFactory::createMaterial(genMatDev);
    GenericObjectDevice *genObjDev =
      &(sceneRawObjectsDevice->objects[objIdx]);    
    objectList[objIdx] = ObjectFactory::createObject(genObjDev, mat);
  }
  *world = new ObjectList(objectList, numObjects);

  printf("GenericMaterialDevice destructor on device\n");
}

void SceneDevice::constructScene(const SceneRawObjectsDevice *sceneRawObjectsDevice,
				 StatusCodes& status)
{
  status = StatusCodes::NoError;
  
  // construct the scene on device
  status = CCE(cudaMalloc((void**)&m_world, sizeof(Object*)));
  if (status != StatusCodes::NoError) {
    return;
  }
  constructScene_kernel<<<1,1>>>(sceneRawObjectsDevice, m_world);
  status = CCE(cudaDeviceSynchronize());
  if (status != StatusCodes::NoError) {
    return;
  }
}
