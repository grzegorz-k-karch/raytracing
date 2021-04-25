#ifndef SCENE_RAW_OBJECTS_H
#define SCENE_RAW_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"
#include "StatusCodes.h"
#include "SceneRawObjectsDevice.cuh"

class SceneRawObjects {
public:
  // default constructor
  SceneRawObjects() {}

  void parseScene(const std::string filepath, StatusCodes& status);

  void setCamera(const Camera& camera) {
    m_camera = std::move(camera);
  }
  void addObject(GenericObject&& obj) {
    m_objects.push_back(std::move(obj));
  }
  void addMaterial(GenericMaterial&& mat) {
    m_materials.push_back(std::move(mat));
  }

  SceneRawObjectsDevice* getObjectsOnDevice(StatusCodes &status) const {
    SceneRawObjectsDevice* sceneRawObjectsDevice = copyToDevice(status);
    return sceneRawObjectsDevice;
  }

private:
  SceneRawObjectsDevice* copyToDevice(StatusCodes &status) const;

  Camera m_camera;
  std::vector<GenericObject> m_objects;
  std::vector<GenericMaterial> m_materials;
};

#endif//SCENE_RAW_OBJECTS_H
