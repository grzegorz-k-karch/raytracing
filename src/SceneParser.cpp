#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
// for checking presense of required objects in the scene
#include <set>
#include <algorithm>
#include <memory>

#include "logging.h"
#include "SceneParser.h"

namespace pt = boost::property_tree;

bool foundAny(const std::string objectType, std::set<std::string>& renderableObjects)
{
  return std::find(renderableObjects.begin(), renderableObjects.end(), objectType)
    != renderableObjects.end();
}

bool checkRequiredObjects(const pt::ptree& sceneTree)
{
  bool cameraPresent = false;
  bool renderableObjectPresent = false;
  std::set<std::string> renderableObjects = {"Sphere",
					     "Mesh"};

  pt::ptree::const_iterator node = sceneTree.begin();
  while ((!cameraPresent || !renderableObjectPresent) &&
	 node != sceneTree.end()) {
    std::string objectType = node->first;
    if (objectType == "Camera") {
      cameraPresent = true;
    }
    if (foundAny(objectType, renderableObjects)) {
      renderableObjectPresent = true;
    }
    node++;
  }
  return cameraPresent && renderableObjectPresent;
}

SceneParser::SceneParser(const std::string filepath,
			 SceneObjects& sceneObjects,
			 StatusCodes& status)
{
  status = StatusCodes::NoError;
  // read XML file
  pt::ptree fileTree;
  try {
    pt::read_xml(filepath, fileTree);
  }
  catch(pt::ptree_error& e) {
    BOOST_LOG_TRIVIAL(error) << e.what();
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

  // get all objects in the scene and put them into appropriate lists
  std::set<std::string> renderableObjects = {"Sphere", "Mesh"};
  std::set<std::string> otherObjects = {"Camera"};
  for (auto& it: sceneTree) {
    std::string objectType = it.first;
    if (objectType == "Camera") {
      sceneObjects.camera = std::make_unique<Camera>(it.second);
    }
    else if (foundAny(objectType, renderableObjects)) {
      sceneObjects.objects.push_back(GenericObject(objectType, it.second));
      // get material for the current oject - one of object's attributes
      pt::ptree::const_assoc_iterator material_it = it.second.find("material");
      if (material_it != it.second.not_found()) {
	pt::ptree material = material_it->second;
	std::string materialType = material.get<std::string>("<xmlattr>.value");
	sceneObjects.materials.push_back(GenericMaterial(materialType, material));
      }
      else {
	status = StatusCodes::SceneError;
	return;
      }
    }
    else {
      BOOST_LOG_TRIVIAL(warning) << "Unknown object " << objectType;
    }
  }
}
