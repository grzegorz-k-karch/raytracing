#ifndef GKK_VEC_H
#define GKK_VEC_H

#include <cmath>
#include <iostream>

class vec4;

class vec3 {

 public:
 vec3() : e{0.f, 0.f, 0.f} {}
 vec3(float x, float y, float z) : e{x, y, z} {}
  inline float x() const { return e[0]; }
  inline float y() const { return e[1]; }
  inline float z() const { return e[2]; }
  inline float r() const { return e[0]; }
  inline float g() const { return e[1]; }
  inline float b() const { return e[2]; }

  inline const vec3 operator+() const { return *this; }
  inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  inline float operator[](int i) const { return e[i]; }
  inline float& operator[](int i) { return e[i]; }

  inline vec3& operator+=(const vec3& v2);
  inline vec3& operator-=(const vec3& v2);
  inline vec3& operator*=(const vec3& v2);
  inline vec3& operator/=(const vec3& v2);
  inline vec3& operator*=(const float t);
  inline vec3& operator/=(const float t);
  
  operator vec4() const;
  friend std::ostream& operator<<(std::ostream& os, const vec3& v);

  float squared_length() const {
    return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
  }
  float length() const { return std::sqrt(squared_length()); }
    
  float e[3];
};

inline vec3& vec3::operator+=(const vec3& v2) {
    e[0] += v2[0];
    e[1] += v2[1];
    e[2] += v2[2];
    return *this;
}

inline vec3 operator+(const vec3& v1, const vec3& v2)
{
  return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

inline vec3 operator-(const vec3& v1, const vec3& v2)
{
  return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

inline vec3 operator*(const vec3& v, const float t)
{
  return vec3(v[0]*t, v[1]*t, v[2]*t);
}

inline vec3 operator*(const float t, const vec3& v)
{
  return vec3(v[0]*t, v[1]*t, v[2]*t);
}

inline vec3 operator*(const vec3& v1, const vec3& v2)
{
  return vec3(v1[0]*v2[0], v1[1]*v2[1], v1[2]*v2[2]);
}

inline vec3 operator/(const vec3& v, const float t)
{
  return vec3(v[0]/t, v[1]/t, v[2]/t);
}

inline float dot(const vec3& v1, const vec3& v2)
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

inline vec3 normalize(const vec3& v)
{
  float norm = std::sqrt(dot(v, v));
  return v/norm;
}

inline vec3 cross(const vec3& v1, const vec3& v2)
{
  return vec3(v1[1]*v2[2] - v1[2]*v2[1],
	      v1[2]*v2[0] - v1[0]*v2[2],
	      v1[0]*v2[1] - v1[1]*v2[0]);
}

inline vec3 reflect(const vec3& v, const vec3& n)
{
  return v- 2.0f*dot(v,n)*n;
}

inline bool refract(const vec3& v, const vec3& n, float n1_over_n2, vec3& refracted)
{
  vec3 uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - n1_over_n2*n1_over_n2*(1.0f - dt*dt);
  if (discriminant > 0.0f) {
    refracted = n1_over_n2*(uv - n*dt) - n*std::sqrt(discriminant);
    return true;
  }
  return false;
}


class vec4 {

 public:
 vec4() : e{0.f, 0.f, 0.f, 1.f} {}
 vec4(float x, float y, float z, float w) : e{x, y, z, w} {}
  inline float x() { return e[0]; }
  inline float y() { return e[1]; }
  inline float z() { return e[2]; }
  inline float w() { return e[3]; }
  inline float r() { return e[0]; }
  inline float g() { return e[1]; }
  inline float b() { return e[2]; }
  inline float a() { return e[3]; }

  inline vec4 operator+() const { return *this; }
  inline vec4 operator-() const { return vec4(-e[0], -e[1], -e[2], -e[3]); }

  operator vec3() const;

  float e[4];
};


#endif//GKK_VEC_H
