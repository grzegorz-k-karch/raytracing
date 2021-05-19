#ifndef SPHERE_LOADER_H
#define SPHERE_LOADER_H

#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

#include "AABB.cuh"

namespace pt = boost::property_tree;

class SphereLoader {
public:
  SphereLoader(const pt::ptree object);
  void loadSphere(AABB &m_bbox,
		  std::vector<float3> &m_vectors,
                  std::vector<float> &m_scalars) const;
private:
  float3 m_center;
  float m_radius;
};

#endif//SPHERE_LOADER_H
