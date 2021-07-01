#ifndef SCENE_RAW_OBJECTS_DEVICE_H
#define SCENE_RAW_OBJECTS_DEVICE_H

// #include "cuda_utils.cuh"

#include "GenericObjectDevice.cuh"
#include "GenericMaterialDevice.cuh"
#include "Camera.cuh"

struct SceneRawObjectsDevice {

  SceneRawObjectsDevice() :
    m_camera(nullptr),
    m_objects(nullptr), m_numObjects(0),
    m_materials(nullptr), m_numMaterials(0) {}

  ~SceneRawObjectsDevice() {
  }
  
  Camera* m_camera;
  GenericObjectDevice* m_objects;
  int m_numObjects;
  GenericMaterialDevice* m_materials;
  int m_numMaterials;
};

#endif//SCENE_RAW_OBJECTS_DEVICE_H
