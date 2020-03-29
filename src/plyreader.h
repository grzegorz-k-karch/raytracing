#ifndef PLYREADER_H
#define PLYREADER_H

#include "ply.h"
#include <vector>
#include <string>

typedef struct {
  unsigned char nindices;
  int *indices;
} polygon_t;

void readPlyObject(const std::string& ply_filename,
		   std::vector<float> &vertices,
		   std::vector<float> &normals,
		   std::vector<unsigned int> &indices);

#endif // PLYREADER_H
