#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "logging.h"
#include "StatusCodes.h"
#include "GenericObjectDevice.cuh"

namespace pt = boost::property_tree;

class GenericObject {
 public:

  GenericObject(const std::string objectType, const pt::ptree object) {
    if (objectType == "Mesh") {
      m_objectType = ObjectType::Mesh;
      parseMesh(object);
    }
    else if (objectType  == "Sphere") {
      m_objectType = ObjectType::Sphere;
      parseSphere(object);
    }
  }

  void copyToDevice(GenericObjectDevice* genericObjectDevice,
		    StatusCodes& status);

private:

  void parseMesh(const pt::ptree object);
  void parseSphere(const pt::ptree object);

  ObjectType m_objectType;
  // sphere members
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
  // mesh members
  std::vector<float3> m_vertices;
  std::vector<float3> m_vertexColors;
  std::vector<float3> m_vertexNormals;
  std::vector<int>    m_triangleIndices;
};

#endif//GENERIC_OBJECT_H
