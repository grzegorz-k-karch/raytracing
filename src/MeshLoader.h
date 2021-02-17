#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <string>
#include <vector>
#include <vector_types.h>

class MeshLoader {
 public:
  MeshLoader() {}
  void loadMesh(const std::string& filepath);

private:
  std::vector<float3> m_vertices;
  std::vector<float3> m_vertex_colors;
  std::vector<float3> m_vertex_normals;
  std::vector<int>    m_triangle_indices;
};

#endif//MESH_LOADER_H
