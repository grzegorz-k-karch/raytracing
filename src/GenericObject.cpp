#include "logging.h"
#include "GenericObject.h"
#include "MeshLoader.h"
#include "SphereLoader.h"

GenericObject::GenericObject(const std::string objectType,
			     const pt::ptree object)
{
  if (objectType == "Mesh") {
    m_objectType = ObjectType::Mesh;
    parseMesh(object);
  }
  else if (objectType  == "Sphere") {
    m_objectType = ObjectType::Sphere;
    parseSphere(object);
  }
  StatusCodes status = StatusCodes::NoError;
  auto material_it = object.find("material");
  bool materialFound = material_it != object.not_found();
  if (materialFound) {
    LOG_TRIVIAL(trace) << "Material found.";
    pt::ptree material = material_it->second;
    m_material = new GenericMaterial(material, status);
  }
  else {
    LOG_TRIVIAL(trace) << "Material not found.";
  }
}

GenericObject::GenericObject(GenericObject&& other) noexcept :
  m_objectType(other.m_objectType),
  m_bbox(other.m_bbox),
  m_material(other.m_material),
  m_scalars(other.m_scalars),
  m_vectors(other.m_vectors),
  m_vertices(other.m_vertices),
  m_vertexColors(other.m_vertexColors),
  m_vertexNormals(other.m_vertexNormals),
  m_textureCoords(other.m_textureCoords),
  m_triangleIndices(other.m_triangleIndices)
{
  LOG_TRIVIAL(trace) << "GenericObject copy constructor";
}

void GenericObject::parseMesh(const pt::ptree object)
{
  MeshLoader meshLoader = MeshLoader(object);
  meshLoader.loadMesh(m_bbox, m_vertices, m_vertexColors,
		      m_vertexNormals, m_textureCoords,
		      m_triangleIndices, m_scalars);
}

void GenericObject::parseSphere(const pt::ptree object)
{
  SphereLoader sphereLoader = SphereLoader(object);
  sphereLoader.loadSphere(m_bbox, m_vectors, m_scalars);
}
