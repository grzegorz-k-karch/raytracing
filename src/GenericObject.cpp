#include "logging.h"
#include "GenericObject.h"
#include "MeshLoader.h"
#include "SphereLoader.h"

void GenericObject::parseMesh(const pt::ptree object)
{
  MeshLoader meshLoader = MeshLoader();
  meshLoader.loadMesh(object);

  m_vertices = meshLoader.getVertices();
  m_vertexColors = meshLoader.getVertexColors();
  m_vertexNormals = meshLoader.getVertexNormals();
  m_triangleIndices = meshLoader.getTriangleIndices();
}

void GenericObject::parseSphere(const pt::ptree object)
{
  SphereLoader sphereLoader = SphereLoader();
  sphereLoader.loadSphere(object);

  m_scalars = sphereLoader.getScalars();
  m_vectors = sphereLoader.getVectors();
}
