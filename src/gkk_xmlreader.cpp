#include "gkk_xmlreader.h"

#include <boost/property_tree/xml_parser.hpp>

void XmlWrite(pt::ptree& tree, const std::string& filename) {
  pt::xml_writer_settings<std::string> settings(' ', 4);
  pt::write_xml(filename, tree, std::locale(), settings);
}
