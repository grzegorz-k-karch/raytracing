#include <string>

#include "SphereLoader.h"
#include "vector_utils.h"
#include "logging.h"

void SphereLoader::loadSphere(const pt::ptree object)
{
  float3 center = string2float3(object.get<std::string>("center.<xmlattr>.value"));
  m_vectors = {center};

  float radius = object.get<float>("radius.<xmlattr>.value");
  m_scalars = {radius};

  LOG_TRIVIAL(debug)
    << "Sphere center: (" << center.x << "," << center.y << "," << center.z << ")";
  LOG_TRIVIAL(debug) << "Sphere radius: " << radius;
}