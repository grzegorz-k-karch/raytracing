#ifndef SCENE_PARSE_H
#define SCENE_PARSE_H

#include <string>
#include <vector>

#include "SceneObjects.h"
#include "StatusCodes.h"

void parseScene(const std::string filepath,
		SceneObjects& sceneObjects,
		StatusCodes& status);
#endif//SCENE_PARSER_H
