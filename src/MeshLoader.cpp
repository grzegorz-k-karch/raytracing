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
  m_translation = string2float3(object.get<std::string>("translation.<xmlattr>.value", "0 0 0"));
  m_scale = string2float3(object.get<std::string>("scale.<xmlattr>.value", "1 1 1"));
  m_rotation = string2float3(object.get<std::string>("rotation.<xmlattr>.value", "0 0 0"));
  m_smoothness = object.get<float>("smoothness.<xmlattr>.value", 1.0f);
  m_frontFace = object.get<std::string>("front_face.<xmlattr>.value", std::string("CCW"));

  LOG_TRIVIAL(debug) << "Mesh filepath: " << m_meshFilepath;
}


bool checkIfFileExists(std::string filepath)
{
  bool fileExists = false;
  std::ifstream file(filepath, std::ios::in);
  if (file.is_open() && file.good()) {
    fileExists = true;
    file.close();
  }
  return fileExists;
}


void MeshLoader::loadMesh(AABB& bbox,
			  std::vector<float3>& vertices,
			  std::vector<float3>& vertexColors,
			  std::vector<float3>& vertexNormals,
			  std::vector<float2>& textureCoords,
			  std::vector<int>& triangleIndices,
			  std::vector<float>& scalars,
			  StatusCode& status) const
{
  bool fileExists = checkIfFileExists(m_meshFilepath);
  if (!fileExists) {
    status = StatusCode::FileError;
    LOG_TRIVIAL(error) << "Could not open " << m_meshFilepath << ".";
    return;
  }
  bool fileIsPly = checkIfPlyFile(m_meshFilepath);
  LOG_TRIVIAL(debug) << "File is PLY: " << fileIsPly;
  if (fileIsPly) {
    loadPlyObject(m_meshFilepath.c_str(), vertices,
		  vertexColors, vertexNormals,
		  textureCoords, triangleIndices, status);
    if (status != StatusCode::NoError) {
      return;
    }
    LOG_TRIVIAL(debug) << "\nNum vertices: " << vertices.size()
		       << "\nnum colors: " << vertexColors.size()
		       << "\nnum normals: " << vertexNormals.size()
		       << "\nnum coords: " << textureCoords.size()
		       << "\nnum indices: " << triangleIndices.size();
  }
  else {
    LOG_TRIVIAL(error) << "Only PLY files are currently supported";
    status = StatusCode::FileError;
    return;
  }

  // cleanup mesh
  //   merge vertices
  // std::vector<int> indicesOfKeptVertices;
  // mergeVertices(triangleIndices, vertices, indicesOfKeptVertices);
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
    // vertexColors = mergeVectors(indicesOfKeptVertices, vertexColors);
  }
  LOG_TRIVIAL(debug) << "After colors merge: num colors: " << vertexColors.size();

  //   compute normals if not present
  if (vertexNormals.empty()) {
    // if no normals - compute them
    computeNormals(vertices, triangleIndices, vertexNormals, m_frontFace);
  }
  else {
    // if there are normals - merge them according to the merged vertices
    // vertexNormals = mergeVectors(indicesOfKeptVertices, vertexNormals);
  }
  LOG_TRIVIAL(debug) << "After normals merge: num normals: " << vertexNormals.size();

  // compute axis-aligned bounding box
  float3 bmin;
  float3 bmax;
  computeBBox(vertices, bmin, bmax);

  LOG_TRIVIAL(debug) << "transforming geometry ...";
  scaleRotateTranslate(m_scale, m_rotation, m_translation,
		       bmin, bmax, vertices, vertexNormals);
  LOG_TRIVIAL(debug) << "transforming geometry done";

  computeBBox(vertices, bmin, bmax);

  bbox = AABB(bmin, bmax);
  scalars.push_back(m_smoothness);
}
