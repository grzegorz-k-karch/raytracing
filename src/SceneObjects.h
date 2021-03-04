#ifndef SCENE_OBJECTS_H
#define SCENE_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.h"

struct SceneObjects {
  std::unique_ptr<Camera> camera;
  std::vector<GenericObject> objects;
  std::vector<GenericMaterial> materials;
};

#endif//SCENE_OBJECTS_H
