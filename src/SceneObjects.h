#ifndef SCENE_OBJECTS_H
#define SCENE_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"

struct SceneObjects {
  std::unique_ptr<Camera> camera;
  std::vector<GenericObject> objects;
  std::vector<GenericMaterial> materials;
};

struct SceneObjectsDevice {
  Camera* camera;
  GenericObjectDevice* objects;
  GenericMaterialDevice* materials;
};

#endif//SCENE_OBJECTS_H
