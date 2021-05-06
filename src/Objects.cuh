#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include "Ray.cuh"
#include "GenericObjectDevice.cuh"

class Material;


class Object {
 public:
  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const = 0;
  __device__ virtual
  bool getBBox(float tMin, float tMax, AABB& bbox) const = 0;  
};


class ObjectList: public Object {
 public:
  __device__ ObjectList(Object** objects, int num_objects) :
    objects(objects), num_objects(num_objects), bboxComputed(false) {}

  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(float tMin, float tMax, AABB& bbox) const;

  Object **objects;
  int num_objects;

  AABB bbox;
  bool bboxComputed;
};


class Mesh : public Object {
public:
  __device__ Mesh(const GenericObjectDevice* genObjDev,
		  const Material* mat)
    : m_bbox(AABB(genObjDev->bmin, genObjDev->bmax)),
      vertices(genObjDev->vertices),
      numVertices(genObjDev->numVertices),
      vertexColors(genObjDev->vertexColors),
      numVertexColors(genObjDev->numVertexColors),
      vertexNormals(genObjDev->vertexNormals),
      numVertexNormals(genObjDev->numVertexNormals),
      triangleIndices(genObjDev->triangleIndices),
      numTriangleIndices(genObjDev->numTriangleIndices),
      m_material(mat) {}

  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(float tMin, float tMax, AABB &bbox) const {
    bbox = m_bbox;
    return true;
  }

  AABB m_bbox;

  float3 *vertices;
  int    numVertices;
  float3 *vertexColors;
  int    numVertexColors;
  float3 *vertexNormals;
  int    numVertexNormals;
  int *triangleIndices;
  int numTriangleIndices;
  const Material *m_material;

private:
  __device__
  float3 normalAtP(float3 point,
		   float3 vert0,
		   float3 vert1,
		   float3 vert2) const;
};


class Sphere : public Object {
public:
  __device__ Sphere(const GenericObjectDevice* genObjDev,
		    const Material* mat)
    : m_bbox(AABB(genObjDev->bmin, genObjDev->bmax)),
      m_center(genObjDev->vectors[0]),
      m_radius(genObjDev->scalars[0]),
      m_material(mat) {}
  
  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(float tMin, float tMax, AABB &bbox) const {
    bbox = m_bbox;
    return true;
  }

  AABB m_bbox;
  float3 m_center;
  float m_radius;
  const Material *m_material;

private:
  __device__
  float3 normalAtP(float3 point) const;
};

#endif//OBJECTS_CUH
