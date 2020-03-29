#ifndef GKK_XML_READER_H
#define GKK_XML_READER_H

#include <string>

#include <boost/property_tree/ptree.hpp>
namespace pt = boost::property_tree;

void XmlWrite(pt::ptree& tree, const std::string& filename);

#endif//GKK_XML_READER_H
