#ifndef SCENE_RAW_OBJECTS_DEVICE_H
#define SCENE_RAW_OBJECTS_DEVICE_H

#include "GenericObjectDevice.cuh"
#include "Camera.cuh"

struct SceneRawObjectsDevice {

  SceneRawObjectsDevice() :
    m_camera(nullptr),
    m_objects(nullptr), m_numObjects(0) {}

  ~SceneRawObjectsDevice() {
  }
  
  Camera* m_camera;
  GenericObjectDevice* m_objects;
  int m_numObjects;
};

#endif//SCENE_RAW_OBJECTS_DEVICE_H
