#include "vector_utils.h"

float3 string2float3(const std::string& s)
{
  float3 v;
  std::istringstream iss(s);
  iss >> v.x >> v.y >> v.z;
  return v;
}

float2 string2float2(const std::string& s)
{
  float2 v;
  std::istringstream iss(s);
  iss >> v.x >> v.y;
  return v;
}
