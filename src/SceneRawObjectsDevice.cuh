#ifndef SCENE_RAW_OBJECTS_DEVICE_H
#define SCENE_RAW_OBJECTS_DEVICE_H

// #include "cuda_utils.cuh"

#include "GenericObjectDevice.cuh"
#include "GenericMaterialDevice.cuh"
#include "Camera.cuh"

struct SceneRawObjectsDevice {

  SceneRawObjectsDevice() :
    camera(nullptr),
    objects(nullptr),
    materials(nullptr) {}

  ~SceneRawObjectsDevice() {
  }
  
  Camera* camera;
  GenericObjectDevice* objects;
  int numObjects;
  GenericMaterialDevice* materials;
  int numMaterials;
};

#endif//SCENE_RAW_OBJECTS_DEVICE_H
