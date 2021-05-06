#ifndef SPHERE_LOADER_H
#define SPHERE_LOADER_H

#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

#include "AABB.cuh"

namespace pt = boost::property_tree;

class SphereLoader {
public:
  SphereLoader() {}
  void loadSphere(const pt::ptree object);

  AABB getBBox() {
    return m_bbox;
  }

  std::vector<float>&& getScalars() {
    return std::move(m_scalars);    
  }
  std::vector<float3>&& getVectors() {
    return std::move(m_vectors);
  }  

private:
  AABB m_bbox;
  std::vector<float3> m_vectors;
  std::vector<float> m_scalars;
};

#endif//SPHERE_LOADER_H
