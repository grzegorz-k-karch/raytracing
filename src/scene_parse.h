#ifndef SCENE_PARSE_H
#define SCENE_PARSE_H

#include <string>
#include <vector>

#include "SceneRawObjects.h"
#include "StatusCodes.h"

void parseScene(const std::string filepath,
		SceneRawObjects& sceneObjects,
		StatusCodes& status);

#endif//SCENE_PARSER_H
