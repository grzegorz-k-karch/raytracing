#ifndef PLY_UTILS_H
#define PLY_UTILS_H

#include <vector>
#include <vector_types.h>
#include <string>

typedef struct {
  unsigned char nindices;
  int *indices;
} polygon_t;

bool checkIfPlyFile(const std::string& filepath);

void loadPlyObject(const char* filepath,
		   std::vector<float3>& vertices,
		   std::vector<float3>& vertex_colors,		   
		   std::vector<float3>& vertex_normals,
		   std::vector<float2>& vertex_coords,
		   std::vector<int>& indices);

#endif // PLY_UTILS_H
