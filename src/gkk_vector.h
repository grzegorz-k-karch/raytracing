#ifndef GKK_VECTOR_H
#define GKK_VECTOR_H

class vec3 {

 public:
 vec3() : e{0.f, 0.f, 0.f} {}
 vec3(float x, float y, float z) : e{x, y, z} {}
  inline float x() { return e[0]; }
  inline float y() { return e[1]; }
  inline float z() { return e[2]; }
  inline float r() { return e[0]; }
  inline float g() { return e[1]; }
  inline float b() { return e[2]; }

  inline vec3 operator+() const { return *this; }
  inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

  inline operator vec4() const { return vec4(e[0], e[1], e[2], 1.0f); }
    
  float e[3];
};

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

  inline operator vec3() const { return vec3(e[0], e[1], e[2]); }
    
  float e[4];
};


#endif//GKK_VECTOR_H
