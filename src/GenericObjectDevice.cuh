#ifndef GENERIC_OBJECT_DEVICE_H
#define GENERIC_OBJECT_DEVICE_H

#include "AABB.cuh"

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  GenericObjectDevice() :
    m_objectType(ObjectType::None),
    m_bmin(make_float3(0.0f, 0.0f, 0.0f)),
    m_bmax(make_float3(0.0f, 0.0f, 0.0f)),
    m_scalars(nullptr), m_numScalars(0),
    m_vectors(nullptr), m_numVectors(0),
    m_vertices(nullptr), m_numVertices(0),
    m_vertexColors(nullptr), m_numVertexColors(0),
    m_vertexNormals(nullptr), m_numVertexNormals(0),
    m_textureCoords(nullptr), m_numTextureCoords(0),
    m_triangleIndices(nullptr), m_numTriangleIndices(0) {}

  ~GenericObjectDevice() {}

  ObjectType m_objectType;
  float3 m_bmin;
  float3 m_bmax;
  // sphere members
  float *m_scalars;
  int m_numScalars;
  float3 *m_vectors;
  int m_numVectors;
  // mesh members
  float3 *m_vertices;
  int m_numVertices;
  float3 *m_vertexColors;
  int m_numVertexColors;
  float3 *m_vertexNormals;
  int m_numVertexNormals;
  float2 *m_textureCoords;
  int m_numTextureCoords;
  int *m_triangleIndices;
  int m_numTriangleIndices;
};

#endif//GENERIC_OBJECT_DEVICE_H
