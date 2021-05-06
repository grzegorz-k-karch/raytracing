#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <string>
#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

#include "AABB.cuh"

namespace pt = boost::property_tree;

/// Class for loading mesh objects from disk

/// Currently, MeshLoader class only supports loading mesh objects from PLY
/// files.
class MeshLoader {
 public:
  MeshLoader(const pt::ptree object);
  void loadMesh();

  AABB getBBox() {
    return m_bbox;
  }

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
  std::string m_meshFilepath;
  float3 m_worldPos;
  float3 m_scale;
  AABB m_bbox;

  std::vector<float3> m_vertices;
  std::vector<float3> m_vertexColors;
  std::vector<float3> m_vertexNormals;
  std::vector<int>    m_triangleIndices;
};

#endif//MESH_LOADER_H
