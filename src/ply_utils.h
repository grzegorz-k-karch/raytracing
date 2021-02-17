#ifndef PLY_UTILS_H
#define PLY_UTILS_H

#include "ply_io.h"

typedef struct {
  unsigned char nindices;
  int *indices;
} polygon_t;

void loadPlyObject(const char* filepath,
		   float** vertices, float** vertex_colors,		   
		   float** vertex_normals, int** indices);

#endif // PLY_UTILS_H
