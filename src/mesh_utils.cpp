#include <map>
#include <vector_functions.h>

#include "mesh_utils.h"
#include "nvidia/helper_math.h"

void setColor(const float3 defaultColor,
	      std::vector<float3>& vertexColors)
{
  for (auto &vc : vertexColors) {
    vc = defaultColor;
  }
}

class VertexComparator {
public:
  bool operator()(const float3& a, const float3& b) const {
    return (a.x < b.x ||
	    (a.x == b.x && (a.y < b.y ||
			    (a.y == b.y && (a.z < b.z)))));
  }
};

void mergeVertices(std::vector<int>& indices,
		   std::vector<float3>& vertices,
		   std::vector<int>& indicesOfKeptVertices)
{    
  std::map<float3, int, VertexComparator> visitedVertices;
  std::vector<float3> mergedVertices;
  std::vector<int> mergedIndices;  
  
  int mergedVertexID = 0;
  for (auto vertexIdx : indices) {

    float3 &key = vertices[vertexIdx];
    if (visitedVertices.find(key) == visitedVertices.end()) {

      visitedVertices[key] = mergedVertexID;

      mergedVertices.push_back(key);
      mergedIndices.push_back(mergedVertexID);
      indicesOfKeptVertices.push_back(vertexIdx);

      mergedVertexID++;
    }
    else {
      mergedIndices.push_back(visitedVertices[key]);
    }
  }
  vertices = mergedVertices;
  indices = mergedIndices;
}


void computeNormals(const std::vector<float3>& vertices,
		    const std::vector<int>& indices,
		    std::vector<float3>& normals)
{
  normals.resize(vertices.size(), make_float3(0.0f));

  int numTriangles = indices.size()/3;
  for (int triaIdx = 0; triaIdx < numTriangles; triaIdx++) {

    int vertIdx0 = indices[triaIdx*3];
    int vertIdx1 = indices[triaIdx*3+1];
    int vertIdx2 = indices[triaIdx*3+2];

    float3 triangleVertices[3] = {vertices[vertIdx0],
				  vertices[vertIdx1],
				  vertices[vertIdx2]};
    
    float3 edge1 = triangleVertices[1] - triangleVertices[0];
    float3 edge2 = triangleVertices[2] - triangleVertices[0];
    float3 n = cross(edge1,edge2);

    normals[vertIdx0] += n;
    normals[vertIdx1] += n;
    normals[vertIdx2] += n;
  }
  for (int normIdx = 0; normIdx < normals.size(); normIdx++) {
    float3 n = normals[normIdx];
    normals[normIdx] = normalize(n);
  }  
}


void computeBBox(const std::vector<float3>& vertices,
		 float3& bmin, float3& bmax)
{
  bmax = vertices[0];
  bmin = vertices[0];
  for (auto& v : vertices) {
    if (v.x < bmin.x) {
      bmin.x = v.x;
    }
    if (v.y < bmin.y) {
      bmin.y = v.y;
    }
    if (v.z < bmin.z) {
      bmin.z = v.z;
    }
    if (bmax.x < v.x) {
      bmax.x = v.x;
    }
    if (bmax.y < v.y) {
      bmax.y = v.y;
    }
    if (bmax.z < v.z) {
      bmax.z = v.z;
    }
  }

}

void translateAndScale(float3 worldPos, float3 scale,
		       float3& bmin, float3& bmax,
		       std::vector<float3>& vertices)
{
  float3 center = make_float3((bmin.x + bmax.x)/2.0f,
			      (bmin.y + bmax.y)/2.0f,
			      (bmin.z + bmax.z)/2.0f);
  for (auto &v : vertices) {
    v = (v - center)*scale + worldPos;
  }
  bmin = (bmin - center)*scale + worldPos;
  bmax = (bmax - center)*scale + worldPos;
}
