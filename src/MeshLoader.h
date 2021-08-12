#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <string>
#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <vector_types.h>

#include "StatusCode.h"
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
		std::vector<uint3>& indexTriplets,
		std::vector<float>& scalars,
		StatusCode& status) const;

private:
  std::string m_meshFilepath;
  float3 m_translation;
  float3 m_scale;
  float3 m_rotation;
  float m_smoothness;
  std::string m_frontFace;
};

#endif//MESH_LOADER_H
