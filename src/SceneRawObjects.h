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
  SceneRawObjects() :
    m_sceneRawObjectsDevice(nullptr) {
    //TODO: do logging
  }

  // TODO: understand move operator
  void setCamera(Camera&& camera) {
    m_camera = std::move(camera);
  }
  void addObject(GenericObject&& obj) {
    m_objects.push_back(std::move(obj));
  }
  void addMaterial(GenericMaterial&& mat) {
    m_materials.push_back(std::move(mat));
  }

  SceneRawObjectsDevice *getObjectsOnDevice() {
    return m_sceneRawObjectsDevice;
  }

  void copyToDevice(StatusCodes &status);

private:
  Camera m_camera;
  std::vector<GenericObject> m_objects;
  std::vector<GenericMaterial> m_materials;
  
  SceneRawObjectsDevice *m_sceneRawObjectsDevice;
};

#endif//SCENE_RAW_OBJECTS_H
