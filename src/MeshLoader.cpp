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
  m_smoothness = object.get<float>("smoothness.<xmlattr>.value", 1.0f);

  LOG_TRIVIAL(debug) << "Mesh filepath: " << m_meshFilepath;
}

void MeshLoader::loadMesh(AABB& bbox,
			  std::vector<float3>& vertices,
			  std::vector<float3>& vertexColors,
			  std::vector<float3>& vertexNormals,
			  std::vector<float2>& vertexCoords,
			  std::vector<int>& triangleIndices,
			  std::vector<float>& scalars) const
{
  bool fileIsPly = checkIfPlyFile(m_meshFilepath);
  LOG_TRIVIAL(debug) << "File is PLY: " << fileIsPly;
  if (fileIsPly) {
    loadPlyObject(m_meshFilepath.c_str(), vertices,
		  vertexColors, vertexNormals,
		  vertexCoords, triangleIndices);
    LOG_TRIVIAL(debug) << "\nNum vertices: " << vertices.size()
		       << "\nnum colors: " << vertexColors.size()
		       << "\nnum normals: " << vertexNormals.size()
		       << "\nnum coords: " << vertexCoords.size()
		       << "\nnum indices: " << triangleIndices.size();
  }

  // cleanup mesh
  //   merge vertices
  std::vector<int> indicesOfKeptVertices;
  mergeVertices(triangleIndices, vertices, indicesOfKeptVertices);
  LOG_TRIVIAL(debug) << "After merging vertices:"
		     << " num vertices: " << vertices.size()
    		     << " num indices: " << triangleIndices.size();

  //   set vertex colors if not present
  if (vertexColors.empty()) {
    // if no colors - set a default color // TODO: get color from scene description file
    vertexColors.resize(vertices.size());
    float3 default_color = make_float3(0.0f, 0.4f, 0.8f);
    setColor(default_color, vertexColors);
  }
  else {
    // if there are colors - merge them according to the merged vertices
    vertexColors = mergeVectors(indicesOfKeptVertices, vertexColors);
  }
  LOG_TRIVIAL(debug) << "After colors cleanup: num colors: " << vertexColors.size();

  //   compute normals if not present
  if (vertexNormals.empty()) {
    // if no normals - compute them
    computeNormals(vertices, triangleIndices, vertexNormals);
  }
  else {
    // if there are normals - merge them according to the merged vertices
    vertexNormals = mergeVectors(indicesOfKeptVertices, vertexNormals);
  }
  LOG_TRIVIAL(debug) << "After normals cleanup: num normals: " << vertexNormals.size();

  //   if vertex coords are present - merge them according to vertex merge
  if (!vertexCoords.empty()) {
    vertexCoords = mergeVectors(indicesOfKeptVertices, vertexCoords);
  }

  // compute axis-aligned bounding box
  float3 bmin;
  float3 bmax;
  computeBBox(vertices, bmin, bmax);

  LOG_TRIVIAL(trace) << "BBox: ("
		     << bmin.x << ", " << bmin.y << ", " << bmin.z << ") - ("
		     << bmax.x << ", " << bmax.y << ", " << bmax.z << ")";

  translateAndScale(m_worldPos, m_scale, bmin, bmax, vertices);

  LOG_TRIVIAL(trace) << "BBox after translation and scaling: ("
		     << bmin.x << ", " << bmin.y << ", " << bmin.z << ") - ("
		     << bmax.x << ", " << bmax.y << ", " << bmax.z << ")";

  bbox = AABB(bmin, bmax);
  scalars.push_back(m_smoothness);
}
