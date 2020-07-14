#ifndef GKK_XML_READER_H
#define GKK_XML_READER_H

#include "gkk_vec.cuh"

#include <string>

#include <boost/property_tree/ptree.hpp>
namespace pt = boost::property_tree;

vec3 string2vec3(const std::string& s);

void xml_read(const std::string& filename, pt::ptree& tree);

#endif//GKK_XML_READER_H
