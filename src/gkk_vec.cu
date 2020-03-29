#include "gkk_vec.cuh"

std::ostream& operator<<(std::ostream& os, const vec3& v)
{
  return os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
}

__device__ vec3::operator vec4() const
{
  return vec4(e[0], e[1], e[2], 1.0f);
}

__device__ vec4::operator vec3() const
{
  return vec3(e[0], e[1], e[2]);
}

