#include <fstream>

#include "ply_io.h"

#include "logging.h"
#include "ply_utils.h"
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

void MeshLoader::loadMesh(const std::string& filepath)
{
  bool fileIsPly = checkIfPlyFile(filepath);
  LOG_TRIVIAL(debug) << "File is PLY: " << fileIsPly;
  if (fileIsPly) {
    float *vertices;
    float *vertex_colors;
    float *vertex_normals;
    int *triangle_indices;
    loadPlyObject(filepath.c_str(),
		  &vertices, &vertex_colors,
		  &vertex_normals, &triangle_indices);    
  }
}


