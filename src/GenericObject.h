#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

#include <vector>
#include <vector_types.h>
#include <boost/property_tree/xml_parser.hpp>

#include "logging.h"
#include "StatusCodes.h"
#include "AABB.cuh"
#include "GenericObjectDevice.cuh"

namespace pt = boost::property_tree;

class GenericObject {
 public:
  // default constructor
  GenericObject() = delete;
  // explicit constructor
  GenericObject(const std::string objectType, const pt::ptree object);
  // move constructor
  GenericObject(GenericObject&& other) noexcept;
  // copy constructor
  GenericObject(const GenericObject& other) = delete;
  // copy assignment operator
  GenericObject& operator=(const GenericObject& other) = delete;
  // move assignment operator
  GenericObject& operator=(const GenericObject&& other) = delete;
  
  void copyToDevice(GenericObjectDevice* genericObjectDevice,
		    StatusCodes& status) const;

private:

  void parseMesh(const pt::ptree object);
  void parseSphere(const pt::ptree object);

  ObjectType m_objectType;

  AABB m_bbox;
  
  // sphere members
  std::vector<float> m_scalars;
  std::vector<float3> m_vectors;
  // mesh members
  std::vector<float3> m_vertices;
  std::vector<float3> m_vertexColors;
  std::vector<float3> m_vertexNormals;
  std::vector<float2> m_vertexCoords;
  std::vector<int>    m_triangleIndices;
};

#endif//GENERIC_OBJECT_H
