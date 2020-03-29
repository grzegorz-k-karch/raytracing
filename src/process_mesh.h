#ifndef PROCESS_MESH_H
#define PROCESS_MESH_H

#include <vector>
#include <map>

int mergeVertices(const std::vector<float> &in_vertices,
                  const std::vector<unsigned> &in_indices,
                  const int vertex_size,
                  std::vector<float> &out_vertices,
                  std::vector<unsigned> &out_indices,
                  std::map<unsigned,unsigned> &merge_map);
void computeNormals(const std::vector<float> &vertices,
                    const std::vector<unsigned> &indices,
                    std::vector<float> &normals);
#endif // PROCESS_MESH_H

