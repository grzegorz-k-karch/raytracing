#ifndef SCENE_RAW_OBJECTS_H
#define SCENE_RAW_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"
#include "StatusCodes.h"

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


class SceneRawObjects {
public:
  // default constructor
  SceneRawObjects() {}
  ~SceneRawObjects();

  void parseScene(const std::string filepath, StatusCodes& status);

  void setCamera(Camera&& camera) {
    m_camera = std::move(camera);
  }
  void addObject(GenericObject&& obj) {
    m_objects.push_back(std::move(obj));
  }

  SceneRawObjectsDevice* getObjectsOnDevice(StatusCodes &status) {
    SceneRawObjectsDevice* sceneRawObjectsDevice = copyToDevice(status);
    return sceneRawObjectsDevice;
  }

private:
  SceneRawObjectsDevice* copyToDevice(StatusCodes &status);

  Camera m_camera;
  std::vector<GenericObject> m_objects;

  SceneRawObjectsDevice m_h_sceneRawObjectsDevice;
};

#endif//SCENE_RAW_OBJECTS_H
