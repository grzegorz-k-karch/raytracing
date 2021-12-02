#include "logging.h"
#include "GenericObject.h"
#include "MeshLoader.h"
#include "SphereLoader.h"

GenericObject::GenericObject(const std::string objectType,
			     const pt::ptree object,
			     StatusCode& status)
{
  if (objectType == "Mesh") {
    m_objectType = ObjectType::Mesh;
    parseMesh(object, status);
  }
  else if (objectType  == "Sphere") {
    m_objectType = ObjectType::Sphere;
    parseSphere(object, status);
  }
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Could not parse "
		       << objectType << " object.";
    return;
  }
}

GenericObject::GenericObject(GenericObject&& other) noexcept :
  m_objectType(other.m_objectType),
  m_bbox(other.m_bbox),
  m_scalars(other.m_scalars),
  m_vectors(other.m_vectors),
  m_vertices(other.m_vertices),
  m_albedo(other.m_albedo),
  m_vertexNormals(other.m_vertexNormals),
  m_textureCoords(other.m_textureCoords),
  m_indexTriplets(other.m_indexTriplets)
{
  LOG_TRIVIAL(trace) << "GenericObject move constructor";
}


GenericObject::~GenericObject()
{
}


void GenericObject::parseMesh(const pt::ptree object,
			      StatusCode& status)
{
  MeshLoader meshLoader = MeshLoader(object);
  meshLoader.loadMesh(m_bbox, m_vertices, m_albedo,
		      m_vertexNormals, m_textureCoords,
		      m_indexTriplets, m_scalars,
		      status);
}

void GenericObject::parseSphere(const pt::ptree object,
				StatusCode& status)
{
  SphereLoader sphereLoader = SphereLoader(object);
  sphereLoader.loadSphere(m_bbox, m_vectors, m_scalars,
			  status);
}
