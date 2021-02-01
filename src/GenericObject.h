#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

namespace pt = boost::property_tree;

enum class ObjectType { None, Mesh, Sphere };

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

private:

  void parseMesh(const pt::ptree object);
  void parseSphere(const pt::ptree object);

  ObjectType m_objectType;
  int m_numScalars;
  float *m_scalars;
  int m_numVectors;
  float3 *m_vectors;
};

#endif//GENERIC_OBJECT_H
