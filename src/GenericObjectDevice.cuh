#ifndef GENERIC_OBJECT_DEVICE_H
#define GENERIC_OBJECT_DEVICE_H

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  ObjectType objectType;

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

  int *triangleIndices;
  int numTriangleIndices;
};

#endif//GENERIC_OBJECT_DEVICE_H
