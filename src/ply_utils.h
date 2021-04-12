#ifndef PLY_UTILS_H
#define PLY_UTILS_H

#include <vector>
#include <vector_types.h>

#include "ply_io.h"

typedef struct {
  unsigned char nindices;
  int *indices;
} polygon_t;

void loadPlyObject(const char* filepath,
		   std::vector<float3>& vertices,
		   std::vector<float3>& vertex_colors,		   
		   std::vector<float3>& vertex_normals,
		   std::vector<int>& indices);

#endif // PLY_UTILS_H
