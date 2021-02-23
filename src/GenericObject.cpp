#include "logging.h"
#include "GenericObject.h"
#include "MeshLoader.h"
#include "SphereLoader.h"

void GenericObject::parseMesh(const pt::ptree object)
{
  // TODO: load mesh
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
}

/*
    float3 lookFrom =
      string2float3(camera.get<std::string>("lookFrom.<xmlattr>.value"));
    float3 lookAt =
      string2float3(camera.get<std::string>("lookAt.<xmlattr>.value"));
    float3 up =
      string2float3(camera.get<std::string>("up.<xmlattr>.value"));
    float fov =
      camera.get<float>("fov.<xmlattr>.value");
    float aperture =
      camera.get<float>("aperture.<xmlattr>.value");
    float focus_distance =
      camera.get<float>("focus_distance.<xmlattr>.value");
    if (focus_distance < 0.0f) {
      focus_distance = length(lookFrom-lookAt);
    }
    float time0 = camera.get<float>("time0.<xmlattr>.value");
    float time1 = camera.get<float>("time1.<xmlattr>.value");
*/
