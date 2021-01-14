#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
// for checking presense of required objects in the scene
#include <set>
#include <algorithm>

#include "SceneParser.h"

namespace pt = boost::property_tree;

bool checkRequiredObjects(const pt::ptree& sceneTree)
{
  bool cameraPresent = false;
  bool renderableObjectPresent = false;
  std::set<std::string> renderableObjects = {"sphere",
					      "triangle_mesh"};

  for (auto& it: sceneTree) {
    if (it.first == "camera") {
      cameraPresent = true;
    }
    if (std::find(renderableObjects.begin(), renderableObjects.end(), it.first)
	!= renderableObjects.end()) {
      renderableObjectPresent = true;
    }
  }
  return cameraPresent && renderableObjectPresent;
}

SceneParser::SceneParser(const std::string filepath, StatusCodes& status)
{
  status = StatusCodes::NoError;
  // read XML file
  pt::ptree fileTree;
  try {
    pt::read_xml(filepath, fileTree);
  }
  catch(pt::ptree_error& e) {
    std::cerr << e.what() << std::endl;
    status = StatusCodes::FileError;
    return;
  }

  pt::ptree sceneTree = fileTree.get_child("scene");
  
  // check if required objects are present in the scene:
  // - camera, a renderable object
  bool requiredObjectsPresent = checkRequiredObjects(sceneTree);
  if (!requiredObjectsPresent) {
    status = StatusCodes::SceneError;
    return;
  }

  
}
