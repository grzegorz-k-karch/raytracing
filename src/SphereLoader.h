#ifndef SPHERE_LOADER_H
#define SPHERE_LOADER_H

#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

namespace pt = boost::property_tree;

class SphereLoader {
public:
  SphereLoader() {}
  void loadSphere(const pt::ptree object);

private:
  float3 m_center;
  float m_radius;
};

#endif//SPHERE_LOADER_H
