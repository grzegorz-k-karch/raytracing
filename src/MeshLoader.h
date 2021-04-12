#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <string>
#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

namespace pt = boost::property_tree;

class MeshLoader {
 public:
  MeshLoader() {}
  void loadMesh(const pt::ptree object);

  std::vector<float3>&& getVertices() {
    return std::move(m_vertices);
  }
  std::vector<float3>&& getVertexColors() {
    return std::move(m_vertexColors);
  }
  std::vector<float3>&& getVertexNormals() {
    return std::move(m_vertexNormals);
  }
  std::vector<int>&& getTriangleIndices() {
    return std::move(m_triangleIndices);
  }
  

private:
  std::vector<float3> m_vertices;
  std::vector<float3> m_vertexColors;
  std::vector<float3> m_vertexNormals;
  std::vector<int>    m_triangleIndices;
};

#endif//MESH_LOADER_H
