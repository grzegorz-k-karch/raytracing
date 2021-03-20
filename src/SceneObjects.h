#ifndef SCENE_OBJECTS_H
#define SCENE_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"
#include "StatusCodes.h"

struct SceneObjectsDevice {
  Camera* camera;
  GenericObjectDevice* objects;
  int numObjects;
  GenericMaterialDevice* materials;
  int numMaterials;
};

class SceneObjects {
 public:
  std::unique_ptr<Camera> m_camera;
  std::vector<GenericObject> m_objects;
  std::vector<GenericMaterial> m_materials;

  void copyToDevice(SceneObjectsDevice** sceneObjectsDevice,
		    StatusCodes& status);
};

#endif//SCENE_OBJECTS_H
