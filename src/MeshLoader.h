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
  void loadMesh(AABB& bbox,
		std::vector<float3>& vertices,
		std::vector<float3>& vertexColors,
		std::vector<float3>& vertexNormals,
		std::vector<float2>& vertexCoords,
		std::vector<int>& triangleIndices,
		std::vector<float>& scalars) const;
  
private:
  std::string m_meshFilepath;
  float3 m_worldPos;
  float3 m_scale;
  float m_smoothness;
  std::string m_frontFace;
};

#endif//MESH_LOADER_H
