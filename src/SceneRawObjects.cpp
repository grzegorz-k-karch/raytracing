#include <boost/property_tree/xml_parser.hpp>
// for checking presense of required objects in the scene
#include <set>
#include <algorithm>
#include <memory>

#include "logging.h"
#include "SceneRawObjects.h"

namespace pt = boost::property_tree;

bool foundAny(const std::string objectType,
	      std::set<std::string> &renderableObjects)
{
  return std::find(renderableObjects.begin(),
		   renderableObjects.end(), objectType) != renderableObjects.end();
}

bool checkRequiredObjects(const pt::ptree &sceneTree)
{
  bool cameraPresent = false;
  bool renderableObjectPresent = false;
  std::set<std::string> renderableObjects = {"Sphere", "Mesh"};
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

void SceneRawObjects::parseScene(const std::string filepath,
				 StatusCodes &status) {
  status = StatusCodes::NoError;

  //----------------------------------------------------------------------------
  // read XML file
  pt::ptree fileTree;
  try {
    pt::read_xml(filepath, fileTree);
  }
  catch (pt::ptree_error &e) {
    LOG_TRIVIAL(error) << e.what();
    status = StatusCodes::FileError;
    return;
  }
  //----------------------------------------------------------------------------
  // check if required objects are present in the scene:
  // - camera, a renderable object
  pt::ptree sceneTree = fileTree.get_child("scene");
  bool requiredObjectsPresent = checkRequiredObjects(sceneTree);
  if (!requiredObjectsPresent) {
    LOG_TRIVIAL(error) << "Scene file does not have required objects "
		       << "(a camera and a renderable object).";    
    status = StatusCodes::SceneError;
    return;
  }

  // get all objects in the scene and put them into appropriate lists
  std::set<std::string> renderableObjects = {"Sphere", "Mesh"};
  std::set<std::string> otherObjects = {"Camera"};

  for (auto &it : sceneTree) {
    std::string objectType = it.first;

    if (objectType == "Camera") {
      setCamera(Camera(it.second));
    } else if (foundAny(objectType, renderableObjects)) {
      // before we store the object, get it's material - one of object's attributes
      auto material_it = it.second.find("material");
      bool materialFound = material_it != it.second.not_found();

      if (materialFound) {
        // we're good to go - store the object
        addObject(GenericObject(objectType, it.second));
        pt::ptree material = material_it->second;
        std::string materialType = material.get<std::string>("<xmlattr>.value");
        // store object's material
        addMaterial(GenericMaterial(materialType, material));
      } else {
        LOG_TRIVIAL(warning) << "No material found for an object of type " << objectType
                             << ". Skipping the object.";
      }
    } else {
      LOG_TRIVIAL(warning) << "Unknown object " << objectType;
    }
  }
}
