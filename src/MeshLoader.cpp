#include <fstream>
#include <vector_functions.h>

#include "ply_io.h"

#include "logging.h"
#include "ply_utils.h"
#include "mesh_utils.h"
#include "MeshLoader.h"


bool checkIfPlyFile(const std::string& filepath)
{
  std::ifstream file(filepath, std::ios::in);
  bool headerIsPly = false;
  if (file.is_open() && file.good()) {
    
    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty()) {
	
  	headerIsPly = line.compare("ply") == 0;
  	if (headerIsPly) {
  	  break;
  	}
      }
    }
  }
  return headerIsPly;
}

void MeshLoader::loadMesh(const pt::ptree object)
{
  std::string filepath = object.get<std::string>("source.<xmlattr>.value");
  LOG_TRIVIAL(debug) << "Mesh filepath: " << filepath;
  
  bool fileIsPly = checkIfPlyFile(filepath);
  LOG_TRIVIAL(debug) << "File is PLY: " << fileIsPly;
  if (fileIsPly) {
    loadPlyObject(filepath.c_str(),
		  m_vertices, m_vertexColors,
		  m_vertexNormals, m_triangleIndices);
    LOG_TRIVIAL(debug) << "Num vertices: " << m_vertices.size()
		       << " num colors: " << m_vertexColors.size()
		       << " num normals: " << m_vertexNormals.size()      
		       << " num indices: " << m_triangleIndices.size();
  }
  
  mergeVertices(m_triangleIndices, m_vertices);
  LOG_TRIVIAL(debug) << "After mergeVertices: num vertices: " << m_vertices.size()
		     << " num indices: " << m_triangleIndices.size();
  // check if vertex colors and normals were loaded
  //   if no colors - set a default color white
  if (m_vertexColors.empty()) {
    m_vertexColors.resize(m_vertices.size());
    float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    setColor(default_color, m_vertexColors);
  }
  LOG_TRIVIAL(debug) << "After setColor: num colors: " << m_vertexColors.size();
  //   if no normals - compute them
  if (m_vertexNormals.empty()) {
    // computeNormals resizes and sets the default value of m_vertexNormals
    computeNormals(m_vertices, m_triangleIndices, m_vertexNormals);
  }
  LOG_TRIVIAL(debug) << "After computeNormals: num normals: " << m_vertexNormals.size();  
}
