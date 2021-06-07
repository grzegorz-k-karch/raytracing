#ifndef MESH_UTILS_H
#define MESH_UTILS_H

#include <vector>
#include <vector_types.h>

void setColor(const float3 defaultColor,
	      std::vector<float3>& vertexColors);

void mergeVertices(std::vector<int>& indices,
		   std::vector<float3>& vertices,
		   std::vector<int>& indicesOfKeptVertices);

template<class T>
void mergeVectors(const std::vector<int>& indicesOfKeptVertices,
		  std::vector<T>& vectors)
{
  std::vector<T> mergedVectors;

  for (auto idx : indicesOfKeptVertices) {
    mergedVectors.push_back(vectors[idx]);
  }
  vectors = mergedVectors;
}



void computeBBox(const std::vector<float3>& vertices,
		 float3& bmin, float3& bmax);

void computeNormals(const std::vector<float3>& vertices,
		    const std::vector<int>& indices,
		    std::vector<float3>& normals);

void translateAndScale(float3 worldPos, float3 scale,
		       float3& bmin, float3& bmax,
		       std::vector<float3>& vertices);

#endif//MESH_UTILS_H
