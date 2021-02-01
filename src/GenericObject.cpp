#include <string>

#include "GenericObject.h"

void GenericObject::parseMesh(const pt::ptree object)
{
  std::string filepath = object.get<std::string>("source.<xmlattr>.value");
  // std::cout << "filepath " << filepath << std::endl;
  // MeshLoader meshLoader = MeshLoader(filepath);
  // meshloader.loadMesh(filepath, m_numScalars, m_scalars, m_numVectors, m_vectors)
    ;
  // int m_numScalars;
  // float *m_scalars;
  // int m_numVectors;
  // float3 *m_vectors;
}

void GenericObject::parseSphere(const pt::ptree object)
{
  
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
