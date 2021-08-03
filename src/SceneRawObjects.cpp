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
  auto node = sceneTree.begin();

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
				 StatusCode &status) {
  status = StatusCode::NoError;

  //----------------------------------------------------------------------------
  // read XML file
  pt::ptree fileTree;
  try {
    pt::read_xml(filepath, fileTree);
  }
  catch (pt::ptree_error &e) {
    LOG_TRIVIAL(error) << e.what();
    status = StatusCode::FileError;
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
    status = StatusCode::SceneError;
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
      addObject(GenericObject(objectType, it.second));
    } else if (objectType != "<xmlcomment>") {
      LOG_TRIVIAL(warning) << "Unknown object " << objectType;
    }
  }
}


SceneRawObjects::~SceneRawObjects()
{
  LOG_TRIVIAL(trace) << "SceneRawObjects destructor";  
}
