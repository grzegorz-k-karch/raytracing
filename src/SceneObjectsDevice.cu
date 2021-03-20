#include "logging.h"
#include "SceneObjects.h"

void SceneObjects::copyToDevice(SceneObjectsDevice** sceneObjectsDevice)
{
  // allocate pointer to sceneObjectsDevice on host -> h_sceneObjectsDevice
  SceneObjectsDevice *h_sceneObjectsDevice = new SceneObjectsDevice;
  
  // copy objects
  int numObjects = m_objects.size();
  h_sceneObjectsDevice->numObjects = numObjects;

  //   allocate buffer for GenericObjectDevice struct
  CCE(cudaMalloc((void**)&(h_sceneObjectsDevice->objects),
		 numObjects*sizeof(GenericObjectDevice)));

  // go over all generic objects and copy them to the allocated
  // buffer h_sceneObjectsDevice->objects
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying object " << objIdx;
    GenericObjectDevice *currGenericObject = &(h_sceneObjectsDevice->objects[objIdx]);
    m_objects[objIdx].copyToDevice(currGenericObject);
  }

  // copy material to device
  int numMaterials = m_materials.size();
  h_sceneObjectsDevice->numMaterials = numMaterials;
  
  //   allocate buffer for GenericMaterialDevice struct
  CCE(cudaMalloc((void**)&(h_sceneObjectsDevice->materials),
		 numMaterials*sizeof(GenericMaterialDevice)));
  
  // go over all generic materials and copy them to the allocated
  // buffer h_sceneObjectsDevice->materials
  for (int objIdx = 0; objIdx < numMaterials; objIdx++) {
    LOG_TRIVIAL(trace) << "Copying material " << objIdx;
    GenericMaterialDevice *currGenericMaterial = &(h_sceneObjectsDevice->materials[objIdx]);
    m_materials[objIdx].copyToDevice(currGenericMaterial);
  }

  // allocate pointer to sceneObjectsDevice on device
  CCE(cudaMalloc((void**)sceneObjectsDevice, sizeof(SceneObjectsDevice)));
  CCE(cudaMemcpy(*sceneObjectsDevice, h_sceneObjectsDevice,
		 sizeof(SceneObjectsDevice), cudaMemcpyHostToDevice));

  delete h_sceneObjectsDevice;
}
