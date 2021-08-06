#include <string>

#include "SphereLoader.h"
#include "nvidia/helper_math.h"
#include "vector_utils.cuh"
#include "logging.h"

SphereLoader::SphereLoader(const pt::ptree object)
{
  m_center = string2float3(object.get<std::string>("center.<xmlattr>.value"));
  m_radius = object.get<float>("radius.<xmlattr>.value");

  LOG_TRIVIAL(debug) << "Sphere center: ("
    << m_center.x << ","
    << m_center.y << ","
    << m_center.z << ")";
  LOG_TRIVIAL(debug) << "Sphere radius: " << m_radius;
}

void SphereLoader::loadSphere(AABB &bbox,
			      std::vector<float3> &vectors,
			      std::vector<float> &scalars,
			      StatusCode& status) const {
  vectors = {m_center};
  scalars = {m_radius};
  bbox = AABB(m_center - make_float3(m_radius),
	      m_center + make_float3(m_radius));  
}

