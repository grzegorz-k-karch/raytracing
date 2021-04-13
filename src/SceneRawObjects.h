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
  
  Camera m_camera;
  std::vector<GenericObject> m_objects;
  std::vector<GenericMaterial> m_materials;

  void copyToDevice(SceneRawObjectsDevice** sceneObjectsDevice,
		    StatusCodes& status);
};

#endif//SCENE_RAW_OBJECTS_H
