#include "gkk_xmlreader.h"
#include "gkk_vec.cuh"
#include <string>
#include <iostream>
#include <boost/property_tree/xml_parser.hpp>

vec3 string2vec3(const std::string& s)
{
  vec3 v;
  std::istringstream iss(s);
  iss >> v[0] >> v[1] >> v[2];
  return v;
}

void xml_read(const std::string& filename, pt::ptree& tree)
{
  pt::read_xml(filename, tree);
  // try {
  // }
  // catch(const pt::ptree_error& e) {
  //   std::cout << "[XML read error] " << e.what() << std::endl;
  // }
}

