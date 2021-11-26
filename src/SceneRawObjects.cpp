#include <boost/property_tree/xml_parser.hpp>
// for checking presense of required objects in the scene
#include <set>
#include <algorithm>
#include <memory>

#include "logging.h"
#include "cuda_utils.cuh"
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


void SceneRawObjects::copyToDevice(std::vector<GenericObject>& objects,
				   StatusCode& status)
{
  status = StatusCode::NoError;
  m_objectsDevice.resize(objects.size());
  for (int objIdx = 0; objIdx < objects.size(); objIdx++) {
    objects[objIdx].copyAttributesToDevice(m_objectsDevice[objIdx], status);
  }
}


void SceneRawObjects::parseScene(const std::string filepath,
				 std::vector<GenericObject>& objects,
				 StatusCode &status)
{
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

  for (auto &it : sceneTree) {
    std::string objectType = it.first;
    if (objectType == "Camera") {
      setCamera(Camera(it.second));
    } else if (foundAny(objectType, renderableObjects)) {
      GenericObject genObj = GenericObject(objectType, it.second, status);
      if (status == StatusCode::NoError) {
	objects.push_back(std::move(genObj));
      }
      else {
	LOG_TRIVIAL(error) << "Could not add an object of type "
			   << objectType << ".";
      }
    } else if (objectType != "<xmlcomment>") {
      LOG_TRIVIAL(warning) << "Unknown object " << objectType;
    }
  }
}


void SceneRawObjects::loadScene(const std::string filepath,
				StatusCode &status)
{
  std::vector<GenericObject> objects;
  parseScene(filepath, objects, status);
  copyToDevice(objects, status);
}


SceneRawObjects::~SceneRawObjects()
{
  LOG_TRIVIAL(trace) << "SceneRawObjects destructor";
}
