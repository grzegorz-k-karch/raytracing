#ifndef GENERIC_OBJECT_DEVICE_H
#define GENERIC_OBJECT_DEVICE_H

#include "AABB.cuh"

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  GenericObjectDevice() :
    scalars(nullptr), numScalars(0),
    vectors(nullptr), numVectors(0),
    vertices(nullptr), numVertices(0),
    vertexColors(nullptr), numVertexColors(0),
    vertexNormals(nullptr), numVertexNormals(0),
    triangleIndices(nullptr), numTriangleIndices(0) {}

  ~GenericObjectDevice() {
    // if (scalars != nullptr) {
    //   CCE(cudaFree(scalars));
    // }
    // if (vectors != nullptr) {
    //   CCE(cudaFree(vectors));
    // }        
    // if (vertices != nullptr) {
    //   CCE(cudaFree(vertices));
    // }        
    // if (vertexColors != nullptr) {
    //   CCE(cudaFree(vertexColors));
    // }        
    // if (vertexNormals != nullptr) {
    //   CCE(cudaFree(vertexNormals));
    // }        
    // if (triangleIndices != nullptr) {
    //   CCE(cudaFree(triangleIndices));
    // }        
  }

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

  int *triangleIndices;
  int numTriangleIndices;
};

#endif//GENERIC_OBJECT_DEVICE_H
