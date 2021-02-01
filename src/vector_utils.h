#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#include <cmath>
#include <string>
#include <sstream>
#include <vector_types.h>
#include <vector_functions.h>

float3 string2float3(const std::string& s);

float2 string2float2(const std::string& s);

inline float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}

inline float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

inline float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float length(float3 v)
{
  return std::sqrt(dot(v, v));
}

inline float3 normalize(float3 v)
{
  float len = length(v);
  if (len == 0.0f) {
    return make_float3(0.0f);
  }
  return v/len;
}

inline float3 cross(float3 a, float3 b)
{
  return make_float3(a.y*b.z - a.z*b.y,
		     a.z*b.x - a.x*b.z,
		     a.x*b.y - a.y*b.x);
}

#endif//VECTOR_UTILS_H
