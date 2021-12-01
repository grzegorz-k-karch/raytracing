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


StatusCode SceneRawObjects::copyToDevice(std::vector<GenericObject>& objects)
{
  StatusCode status = StatusCode::NoError;
  m_objectsDevice.resize(objects.size());
  for (int objIdx = 0; objIdx < objects.size(); objIdx++) {
    status = objects[objIdx].copyAttributesToDevice(m_objectsDevice[objIdx]);
  }
  return StatusCode::NoError;
}


StatusCode SceneRawObjects::parseScene(const std::string filepath,
				       std::vector<GenericObject>& objects)
{
  //----------------------------------------------------------------------------
  // read XML file
  pt::ptree fileTree;
  try {
    pt::read_xml(filepath, fileTree);
  }
  catch (pt::ptree_error &e) {
    LOG_TRIVIAL(error) << e.what();
    return StatusCode::FileError;
  }
  //----------------------------------------------------------------------------
  // check if required objects are present in the scene:
  // - camera, a renderable object
  pt::ptree sceneTree = fileTree.get_child("scene");
  bool requiredObjectsPresent = checkRequiredObjects(sceneTree);
  if (!requiredObjectsPresent) {
    LOG_TRIVIAL(error) << "Scene file does not have required objects "
		       << "(a camera and a renderable object).";
    return StatusCode::SceneError;
  }

  // get all objects in the scene and put them into appropriate lists
  std::set<std::string> renderableObjects = {"Sphere", "Mesh"};

  StatusCode status = StatusCode::NoError;

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
	return status;
      }
    } else if (objectType != "<xmlcomment>") {
      LOG_TRIVIAL(warning) << "Unknown object " << objectType;
    }
  }
  return status;
}


StatusCode SceneRawObjects::loadScene(const std::string filepath)
{
  StatusCode status = StatusCode::NoError;
  std::vector<GenericObject> objects;
  status = parseScene(filepath, objects);
  if (status != StatusCode::NoError) {
    return status;
  }
  status = copyToDevice(objects);
  if (status != StatusCode::NoError) {
    return status;
  }
  return status;
}


SceneRawObjects::~SceneRawObjects()
{
  LOG_TRIVIAL(trace) << "SceneRawObjects destructor";
}
