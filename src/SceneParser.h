#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <string>
#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.h"
#include "StatusCodes.h"

struct SceneObjects {
  std::unique_ptr<Camera> camera;
  std::vector<GenericObject> objects;
  std::vector<GenericMaterial> materials;
  int numObjects;
  int numMaterials;
};

class SceneParser {
 public:
  SceneParser(const std::string filepath,
	      SceneObjects& sceneObjects,
	      StatusCodes& status);
};
#endif//SCENE_PARSER_H
