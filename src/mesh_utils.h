#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include <vector>
#include <vector_types.h>

void setColor(const float3 defaultColor,
	      std::vector<float3>& vertexColors);

void mergeVertices(std::vector<int>& indices,
		   std::vector<float3>& vertices);

void computeNormals(const std::vector<float3>& vertices,
		    const std::vector<int>& indices,
		    std::vector<float3>& normals);

#endif//MESH_UTILS_H
