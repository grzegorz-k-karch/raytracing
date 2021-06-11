#ifndef GENERIC_OBJECT_DEVICE_H
#define GENERIC_OBJECT_DEVICE_H

#include "AABB.cuh"

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  GenericObjectDevice() :
    bmin(make_float3(0.0f, 0.0f, 0.0f)),
    bmax(make_float3(0.0f, 0.0f, 0.0f)),
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0),
    vertices(nullptr), numVertices(0),
    vertexColors(nullptr), numVertexColors(0),
    vertexNormals(nullptr), numVertexNormals(0),
    textureCoords(nullptr), numTextureCoords(0),
    triangleIndices(nullptr), numTriangleIndices(0) {}

  ~GenericObjectDevice() {}

  ObjectType objectType;

  float3 bmin;
  float3 bmax;

  // sphere members
  float *scalars;
  int   numScalars;

  float3 *vectors;
  int    numVectors;

  // mesh members
  float3 *vertices;
  int    numVertices;

  float3 *vertexColors;
  int    numVertexColors;

  float3 *vertexNormals;
  int    numVertexNormals;

  float2 *textureCoords;
  int    numTextureCoords;

  int *triangleIndices;
  int numTriangleIndices;
};

#endif//GENERIC_OBJECT_DEVICE_H
