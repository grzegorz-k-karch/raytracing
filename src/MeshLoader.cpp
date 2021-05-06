#include <fstream>
#include <vector_functions.h>

#include "logging.h"
#include "ply_utils.h"
#include "mesh_utils.h"
#include "MeshLoader.h"
#include "vector_utils.cuh"

MeshLoader::MeshLoader(const pt::ptree object)
{
  m_meshFilepath = object.get<std::string>("source.<xmlattr>.value");
  m_worldPos = string2float3(object.get<std::string>("world_pos.<xmlattr>.value"));
  m_scale = string2float3(object.get<std::string>("scale.<xmlattr>.value"));
  LOG_TRIVIAL(debug) << "Mesh filepath: " << m_meshFilepath;
}

void MeshLoader::loadMesh()
{
  bool fileIsPly = checkIfPlyFile(m_meshFilepath);
  LOG_TRIVIAL(debug) << "File is PLY: " << fileIsPly;
  if (fileIsPly) {
    loadPlyObject(m_meshFilepath.c_str(),
		  m_vertices, m_vertexColors,
		  m_vertexNormals, m_triangleIndices);
    LOG_TRIVIAL(debug) << "Num vertices: " << m_vertices.size()
		       << " num colors: " << m_vertexColors.size()
		       << " num normals: " << m_vertexNormals.size()
		       << " num indices: " << m_triangleIndices.size();
  }

  // cleanup mesh
  //   merge vertices
  std::vector<int> indicesOfKeptVertices;
  mergeVertices(m_triangleIndices, m_vertices, indicesOfKeptVertices);
  LOG_TRIVIAL(debug) << "After merging vertices:"
		     << " num vertices: " << m_vertices.size()
    		     << " num indices: " << m_triangleIndices.size();

  //   set vertex colors if not present
  if (m_vertexColors.empty()) {
    // if no colors - set a default color // TODO: get color from scene description file
    m_vertexColors.resize(m_vertices.size());
    float3 default_color = make_float3(0.0f, 0.4f, 0.8f);
    setColor(default_color, m_vertexColors);
  }
  else {
    // if there are colors - merge them according to the merged vertices
    mergeVectors(indicesOfKeptVertices, m_vertexColors);
  }
  LOG_TRIVIAL(debug) << "After colors cleanup: num colors: " << m_vertexColors.size();

  //   compute normals if not present
  if (m_vertexNormals.empty()) {
    // if no normals - compute them
    computeNormals(m_vertices, m_triangleIndices, m_vertexNormals);
  }
  else {
    // if there are normals - merge them according to the merged vertices
    mergeVectors(indicesOfKeptVertices, m_vertexNormals);
  }
  LOG_TRIVIAL(debug) << "After normals cleanup: num normals: " << m_vertexNormals.size();

  // compute axis-aligned bounding box
  float3 bmin;
  float3 bmax;
  computeBBox(m_vertices, bmin, bmax);

  LOG_TRIVIAL(debug) << "BBox: ("
		     << bmin.x << ", " << bmin.y << ", " << bmin.z << ") - ("
		     << bmax.x << ", " << bmax.y << ", " << bmax.z << ")";

  translateAndScale(m_worldPos, m_scale, bmin, bmax, m_vertices);

  LOG_TRIVIAL(debug) << "BBox after translation and scaling: ("
		     << bmin.x << ", " << bmin.y << ", " << bmin.z << ") - ("
		     << bmax.x << ", " << bmax.y << ", " << bmax.z << ")";

  m_bbox = AABB(bmin, bmax);
}
