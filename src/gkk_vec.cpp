#include "gkk_vec.h"

std::ostream& operator<<(std::ostream& os, const vec3& v)
{
  os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
}

vec3::operator vec4() const
{
  return vec4(e[0], e[1], e[2], 1.0f);
}

vec4::operator vec3() const
{
  return vec3(e[0], e[1], e[2]);
}

