#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include <optix.h>
// #include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "logging.h"
#include "StatusCode.h"
#include "AABB.cuh"

namespace pt = boost::property_tree;

enum class ObjectType { None, Mesh, Sphere };

struct GenericObjectDevice {

  GenericObjectDevice() :
    m_objectType(ObjectType::None),
    m_bmin(make_float3(0.0f, 0.0f, 0.0f)),
    m_bmax(make_float3(0.0f, 0.0f, 0.0f)),
    m_scalars(nullptr), m_numScalars(0),
    m_vectors(nullptr), m_numVectors(0),
    m_vertices(0), m_numVertices(0),
    m_albedo(make_float3(0.0f, 0.0f, 0.0f)),
    m_vertexNormals(nullptr), m_numVertexNormals(0),
    m_textureCoords(nullptr), m_numTextureCoords(0),
    m_indexTriplets(0), m_numIndexTriplets(0) {}

  GenericObjectDevice(GenericObjectDevice&& other) noexcept;
  ~GenericObjectDevice();

  ObjectType m_objectType;
  float3 m_bmin;
  float3 m_bmax;
  // sphere members
  float *m_scalars;
  int m_numScalars;
  float3 *m_vectors;
  int m_numVectors;
  // mesh members
  CUdeviceptr m_vertices;
  int m_numVertices;
  float3 m_albedo;
  float3 *m_vertexNormals;
  int m_numVertexNormals;
  float2 *m_textureCoords;
  int m_numTextureCoords;
  CUdeviceptr m_indexTriplets;
  int m_numIndexTriplets;
};


class GenericObject {
public:
  GenericObject() = delete;  // default constructor
  GenericObject(const std::string objectType, const pt::ptree object,
		StatusCode& status);  // explicit constructor
  GenericObject(GenericObject&& other) noexcept;  // move constructor
  GenericObject(const GenericObject& other) = delete;  // copy constructor
  GenericObject& operator=(const GenericObject& other) = delete;  // copy assignment operator
  GenericObject& operator=(const GenericObject&& other) = delete;  // move assignment operator

  ~GenericObject();

  StatusCode copyAttributesToDevice(GenericObjectDevice& h_genericObjectDevice);
  ObjectType getObjectType() {
    return m_objectType;
  }

private:

  void parseMesh(const pt::ptree object,
		 StatusCode& status);
  void parseSphere(const pt::ptree object,
		   StatusCode& status);

  ObjectType m_objectType;
  AABB m_bbox;
  // sphere members
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
  // mesh members
  std::vector<float3> m_vertices;
  float3 m_albedo;
  std::vector<float3> m_vertexNormals;
  std::vector<float2> m_textureCoords;
  std::vector<uint3>  m_indexTriplets;

  CUdeviceptr m_d_indexTriplets;
  CUdeviceptr m_d_vertices;
};

#endif//GENERIC_OBJECT_H
