#include "gkk_xmlreader.h"
#include "gkk_vec.cuh"
#include <string>
#include <iostream>
#include <boost/property_tree/xml_parser.hpp>

vec3 string_to_vec3(const std::string& s)
{
  vec3 v;
  std::istringstream iss(s);
  iss >> v[0] >> v[1] >> v[2];
  return v;
}

void xml_read(const std::string& filename, pt::ptree& tree)
{
  pt::read_xml(filename, tree);
  try {
    pt::ptree camera = tree.get_child("scene.camera");
    // vec3 lookFrom = camera.get<std::string>("lookFrom.<xmlattr>.value");
    // vec3 lookAt = camera.get<std::string>("lookAt.<xmlattr>.value");
    // vec3 up = camera.get<std::string>("up.<xmlattr>.value");
    float fov = camera.get<float>("fov.<xmlattr>.value");
    float aspect = camera.get<float>("fov.<xmlattr>.value");
    float aperture = camera.get<float>("fov.<xmlattr>.value");
    float focus_distance = camera.get<float>("fov.<xmlattr>.value");
    float time0 = camera.get<float>("time0.<xmlattr>.value");
    float time1 = camera.get<float>("time1.<xmlattr>.value");
    std::cout << "||||> " << fov << std::endl;
    std::cout << "||||> " << time0 << std::endl;
  }
  catch(const pt::ptree_error& e) {
    std::cout << "|||| " << e.what() << std::endl;
  }
}
