#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <string>

#include "StatusCodes.h"

class SceneParser {
 public:
  SceneParser(const std::string filepath, StatusCodes& status);
};

#endif//SCENE_PARSER_H
