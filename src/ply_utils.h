#ifndef PLY_UTILS_H
#define PLY_UTILS_H

#include <vector>
#include <vector_types.h>
#include <string>

#include "StatusCode.h"

typedef struct {
  unsigned char nindices;
  int *indices;
} polygon_t;

bool checkIfPlyFile(const std::string& filepath);

void loadPlyObject(const char* filepath,
		   std::vector<float3>& vertices,
		   std::vector<float3>& vertexColors,
		   std::vector<float3>& vertexNormals,
		   std::vector<float2>& textureCoords,
		   std::vector<int>& triangleIndices,
		   StatusCode& status);

#endif // PLY_UTILS_H
