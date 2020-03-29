#include "process_mesh.h"
#include <map>
#include <iostream>
#include <cmath>

typedef struct {
  float x, y, z;
} Vertex_t;

class compareVertices {
public:
  bool operator()(const Vertex_t &a, const Vertex_t &b) const
  {
    return (a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && (a.z < b.z)))));
  }
};

Vertex_t make_Vertex_t(float x, float y, float z)
{
  Vertex_t v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

Vertex_t operator-(Vertex_t a, Vertex_t b)
{
  return make_Vertex_t(a.x - b.x, a.y - b.y, a.z - b.z);
}
Vertex_t cross(Vertex_t a, Vertex_t b)
{
  return make_Vertex_t(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

float length(Vertex_t a)
{
  return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

Vertex_t normalize(Vertex_t a)
{
  float l = length(a);
  if (l == 0.0f)
    return make_Vertex_t(0.0f, 0.0f, 0.0f);
  return make_Vertex_t(a.x/l, a.y/l, a.z/l);
}

int mergeVertices(const std::vector<float> &in_vertices,
                  const std::vector<unsigned> &in_indices,
                  const int vertex_size,
                  std::vector<float> &out_vertices,
                  std::vector<unsigned> &out_indices,
                  std::map<unsigned,unsigned> &merge_map)
{    
  std::map<Vertex_t, unsigned, compareVertices> vertexInfo;
  int numTriangles = in_indices.size()/3;

  int in_vertexID = 0;
  int out_vertexID = 0;

  for (int t = 0; t < numTriangles; ++t) {

    int va_idx = in_indices[t*3]*vertex_size;
    int vb_idx = in_indices[t*3+1]*vertex_size;
    int vc_idx = in_indices[t*3+2]*vertex_size;

    Vertex_t vrt[3] = {{in_vertices[va_idx], in_vertices[va_idx+1], in_vertices[va_idx+2]},
		       {in_vertices[vb_idx], in_vertices[vb_idx+1], in_vertices[vb_idx+2]},
		       {in_vertices[vc_idx], in_vertices[vc_idx+1], in_vertices[vc_idx+2]}};

    for (int v = 0; v < 3; v++) {

      Vertex_t &key = vrt[v];

      if (vertexInfo.find(key) == vertexInfo.end()) {

	vertexInfo[key] = out_vertexID;

	out_vertices.push_back(key.x);
	out_vertices.push_back(key.y);
	out_vertices.push_back(key.z);

	merge_map[in_vertexID] = out_vertexID;
	out_indices.push_back(out_vertexID);

	out_vertexID++;
      }
      else {
	merge_map[in_vertexID] = vertexInfo[key];
	out_indices.push_back(vertexInfo[key]);
      }
      ++in_vertexID;
    }
  }
  return out_vertexID;
}


void computeNormals(const std::vector<float> &vertices,
                    const std::vector<unsigned> &indices,
                    std::vector<float> &normals)
{
  normals.resize(vertices.size(), 0.0f);

  int numTriangles = indices.size()/3;
  for (int i = 0; i < numTriangles; ++i) {

    int vidx0 = indices[i*3+0];
    int vidx1 = indices[i*3+1];
    int vidx2 = indices[i*3+2];
    Vertex_t v0 = {vertices[vidx0*3+0], vertices[vidx0*3+1], vertices[vidx0*3+2]};
    Vertex_t v1 = {vertices[vidx1*3+0], vertices[vidx1*3+1], vertices[vidx1*3+2]};
    Vertex_t v2 = {vertices[vidx2*3+0], vertices[vidx2*3+1], vertices[vidx2*3+2]};

    Vertex_t e1 = v1 - v0;
    Vertex_t e2 = v2 - v0;
    Vertex_t cr = cross(e1,e2);

    normals[vidx0*3+0] += cr.x;
    normals[vidx0*3+1] += cr.y;
    normals[vidx0*3+2] += cr.z;

    normals[vidx1*3+0] += cr.x;
    normals[vidx1*3+1] += cr.y;
    normals[vidx1*3+2] += cr.z;

    normals[vidx2*3+0] += cr.x;
    normals[vidx2*3+1] += cr.y;
    normals[vidx2*3+2] += cr.z;
  }
  for (int i = 0; i < normals.size()/4; ++i) {
    Vertex_t n = {normals[i*3], normals[i*3+1], normals[i*3+2]};
    n = normalize(n);
  }
}
