#include <string>

#include "SphereLoader.h"
#include "nvidia/helper_math.h"
#include "vector_utils.cuh"
#include "logging.h"

void SphereLoader::loadSphere(const pt::ptree object)
{
  float3 center = string2float3(object.get<std::string>("center.<xmlattr>.value"));
  m_vectors = {center};

  float radius = object.get<float>("radius.<xmlattr>.value");
  m_scalars = {radius};

  m_bbox = AABB(center - make_float3(radius),
		center + make_float3(radius));

  LOG_TRIVIAL(debug)
    << "Sphere center: (" << center.x << "," << center.y << "," << center.z << ")";
  LOG_TRIVIAL(debug) << "Sphere radius: " << radius;
}
