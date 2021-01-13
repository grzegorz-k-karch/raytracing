#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
// for checking presense of required objects in the scene
#include <set>
#include <algorithm>

#include "SceneParser.h"

namespace pt = boost::property_tree;

SceneParser::SceneParser(const std::string filepath)
{
  // read XML file
  pt::ptree scene_tree;
  try {
    pt::read_xml(filepath, scene_tree);
  }
  catch(pt::ptree_error& e) {
    std::cerr << e.what() << std::endl;
  }
  
  // check if required objects are present in the scene:
  // - camera
  // - renderable object
  // TODO put this test in a function
  bool camera_present = false;
  bool renderable_object_present = false;
  std::set<std::string> renderable_objects = {"sphere", "triangle_mesh"};
  for (auto& it: scene_tree) {
    for (auto& it2: it.second) {
      std::cout << "tree: " << it2.first << std::endl;
      if (it2.first == "camera") {
  	camera_present = true;
      }
      if (std::find(renderable_objects.begin(), renderable_objects.end(), it2.first) != renderable_objects.end()) {
	renderable_object_present = true;
      }
    }
  }
  if (camera_present && renderable_object_present) {
    std::cout << "required objects are present in the scene" << std::endl;
  }
}
