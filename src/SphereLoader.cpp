#include <string>

#include "SphereLoader.h"
#include "vector_utils.h"
#include "logging.h"

void SphereLoader::loadSphere(const pt::ptree object)
{
  m_center = string2float3(object.get<std::string>("center.<xmlattr>.value"));
  m_radius = object.get<float>("radius.<xmlattr>.value");
  
  LOG_TRIVIAL(debug)
    << "Sphere center: (" << m_center.x << "," << m_center.y << "," << m_center.z << ")";
  LOG_TRIVIAL(debug) << "Sphere radius: " << m_radius;
}
