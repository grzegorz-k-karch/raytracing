#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "logging.h"

namespace pt = boost::property_tree;

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  ObjectType objectType;
  // sphere members
  float  *scalars;
  int    numScalars;
  float3 *vectors;
  int    numVectors;
  // mesh members
  float3 *vertices;
  int    numVertices;
  float3 *vertexColors;
  int    numVertexColors;
  float3 *vertexNormals;
  int    numVertexNormals;
  int    *triangleIndices;  
  int    numTriangleIndices;  
};


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

  const float*  getScalarsPtr()         { return m_scalars.data(); }
  const float3* getVectorsPtr()         { return m_vectors.data(); }

  const float3* getVerticesPtr()        { return m_vertices.data(); }
  const float3* getColorsPtr()          { return m_vertexColors.data(); }
  const float3* getNormalsPtr()         { return m_vertexNormals.data(); }
  const int*    getTriangleIndicesPtr() { return m_triangleIndices.data(); }

  void copyToDevice(GenericObjectDevice* genericObjectDevice);
  
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
