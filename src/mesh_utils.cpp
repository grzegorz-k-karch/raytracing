#include <map>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector_functions.h>

#include "logging.h"

#include "mesh_utils.h"
#include "nvidia/helper_math.h"


inline float degToRad(float a)
{
  return a*M_PI/180.0f;
}


inline float radToDeg(float a)
{
  return a*180.0/M_PI;
}


typedef struct {
  float e[4][4];
} mat4x4;

typedef struct {
  float e[3][3];
} mat3x3;

mat4x4 identityMat4x4()
{
  mat4x4 I;
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      if (r == c) {
        I.e[r][c] = 1.0f;
      } else {
	I.e[r][c] = 0.0f;
      }
    }
  }
  return I;
}


mat4x4 scaleMat4x4(float3 scale)
{
  mat4x4 S = identityMat4x4();
  S.e[0][0] = scale.x;
  S.e[1][1] = scale.y;
  S.e[2][2] = scale.z;
  return S;
}


mat4x4 translationMat4x4(float3 translation)
{
  mat4x4 T = identityMat4x4();
  T.e[0][3] = translation.x;
  T.e[1][3] = translation.y;
  T.e[2][3] = translation.z;
  return T;
}


mat4x4 operator*(const mat4x4& a, const mat4x4& b)
{
  mat4x4 c;
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      c.e[row][col] = 0.0f;
      for (int i = 0; i < 4; i++) {
	c.e[row][col] += a.e[row][i] * b.e[i][col];
      }
    }
  }
  return c;
}


mat4x4 rotationMat4x4(float3 rotation)
{
  float rotXrad = degToRad(rotation.x);
  float cosX = std::cos(rotXrad);
  float sinX = std::sin(rotXrad);
  mat4x4 Rx = identityMat4x4();
  Rx.e[1][1] = Rx.e[2][2] = cosX;
  Rx.e[1][2] = -sinX;
  Rx.e[2][1] = sinX;

  float rotYrad = degToRad(rotation.y);
  float cosY = std::cos(rotYrad);
  float sinY = std::sin(rotYrad);
  mat4x4 Ry = identityMat4x4();
  Ry.e[0][0] = Ry.e[2][2] = cosY;
  Ry.e[0][2] = sinY;
  Ry.e[2][0] = -sinY;

  float rotZrad = degToRad(rotation.z);
  float cosZ = std::cos(rotZrad);
  float sinZ = std::sin(rotZrad);
  mat4x4 Rz = identityMat4x4();
  Rz.e[0][0] = Rz.e[1][1] = cosZ;
  Rz.e[0][1] = -sinZ;
  Rz.e[1][0] = sinZ;

  mat4x4 R = Rz*Rx*Ry;
  return R;
}


float4 operator*(const mat4x4& M, const float4& v)
{
  float x[4] = {v.x, v.y, v.z, v.w};
  float y[4];
  for (int row = 0; row < 4; row++) {
    y[row] = 0.0f;
    for (int i = 0; i < 4; i++) {
      y[row] += M.e[row][i] * x[i];
    }
  }
  return make_float4(y[0], y[1], y[2], y[3]);
}


float4 operator*(const float4& v, const mat4x4& M)
{
  float x[4] = {v.x, v.y, v.z, v.w};
  float y[4];
  for (int col = 0; col < 4; col++) {
    y[col] = 0.0f;
    for (int i = 0; i < 4; i++) {
      y[col] += x[i]*M.e[i][col];
    }
  }
  return make_float4(y[0], y[1], y[2], y[3]);
}


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
		    const std::vector<uint3>& indices,
		    std::vector<float3>& normals,
		    const std::string frontFace)
{
  normals.resize(vertices.size(), make_float3(0.0f));

  int numTriangles = indices.size()/3;
  for (auto triplet : indices) {

    float3 triangleVertices[3] = {vertices[triplet.x],
				  vertices[triplet.y],
				  vertices[triplet.z]};

    float3 edge1 = triangleVertices[1] - triangleVertices[0];
    float3 edge2 = triangleVertices[2] - triangleVertices[0];
    float3 n = cross(edge1,edge2);

    normals[triplet.x] += n;
    normals[triplet.y] += n;
    normals[triplet.z] += n;
  }
  float orientation = frontFace == "CCW" ? 1.0f : -1.0f;
  for (int normIdx = 0; normIdx < normals.size(); normIdx++) {
    float3 n = normals[normIdx];
    normals[normIdx] = orientation*normalize(n);
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
  const float eps = 0.001f;
  if (std::abs(bmax.x - bmin.x) < eps) {
    bmax.x += eps;
    bmin.x -= eps;
  }
  if (std::abs(bmax.y - bmin.y) < eps) {
    bmax.y += eps;
    bmin.y -= eps;
  }
  if (std::abs(bmax.z - bmin.z) < eps) {
    bmax.z += eps;
    bmin.z -= eps;
  }
  LOG_TRIVIAL(trace) << "BBox: ("
		     << bmin.x << ", " << bmin.y << ", " << bmin.z << ") - ("
		     << bmax.x << ", " << bmax.y << ", " << bmax.z << ")";
}


void scaleRotateTranslate(float3 scale, float3 rotation, float3 translation,
			  float3& bmin, float3& bmax,
			  std::vector<float3>& vertices,
			  std::vector<float3>& normals)
{
  mat4x4 scaleMat = scaleMat4x4(scale);
  mat4x4 rotationMat = rotationMat4x4(rotation);
  mat4x4 translationMat = translationMat4x4(translation);

  mat4x4 transformVertices = translationMat*rotationMat*scaleMat;
  for (auto &v : vertices) {
    float4 t = transformVertices*make_float4(v.x, v.y, v.z, 1.0f);
    v = make_float3(t.x, t.y, t.z);
  }

  mat4x4 invScaleMat = scaleMat4x4(scale*-1.0f);
  mat4x4 transformNormals = rotationMat*invScaleMat;
  for (auto &n : normals) {
    float4 t = make_float4(n.x, n.y, n.z, 0.0f)*transformNormals;
    n = normalize(make_float3(t.x, t.y, t.z));
  }
}
