#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include "Ray.cuh"
#include "GenericObjectDevice.cuh"

class Material;


class Object {
public:
  __device__ virtual bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const = 0;
  __device__ virtual bool getBBox(AABB &bbox) const = 0;
};


class BVHNode : public Object {
public:
  __device__ BVHNode(Object** objects, int start, int end,
  		     curandState* randState);
  __device__ BVHNode(Object *left, Object *right) {
    setChildren(left, right);
  }
  __device__ BVHNode() :
    m_left(nullptr), m_right(nullptr), m_bboxComputed(false) {}

  __device__ virtual bool hit(const Ray& ray,
			      float tMin,
			      float tMax,
			      HitRecord& hitRec) const;
  __device__ virtual bool getBBox(AABB& outBBox) const;
  __device__ void setChildren(Object* left, Object* right);

public:
  Object *m_left;
  Object *m_right;

  AABB m_bbox;
  bool m_bboxComputed;
};


__device__ Object* createBVH(Object** objects, int numObjects);

__device__ bool compareBBoxes(Object* a, Object* b, int axis);


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
      textureCoords(genObjDev->textureCoords),
      numTextureCoords(genObjDev->numTextureCoords),
      triangleIndices(genObjDev->triangleIndices),
      numTriangleIndices(genObjDev->numTriangleIndices),
      m_material(mat),
      m_smoothness(genObjDev->scalars[0]) {}

  __device__ virtual bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;

  __device__ virtual bool getBBox(AABB &bbox) const {
    bbox = m_bbox;
    return true;
  }

  __device__ void getTextureUV(float uTriangle, float vTriangle,
			       int t0, int t1, int t2,
			       float& uTexture, float &vTexture) const;

  AABB m_bbox;

  float3 *vertices;
  int    numVertices;
  float3 *vertexColors;
  int    numVertexColors;
  float3 *vertexNormals;
  int    numVertexNormals;
  float2 *textureCoords;
  int    numTextureCoords;
  int *triangleIndices;
  int numTriangleIndices;
  const Material *m_material;
  float m_smoothness;

private:
  __device__ float3 normalAtP(float u, float v,
			      float3 n0, float3 n1, float3 n2) const;
};


class Sphere : public Object {
public:
  __device__ Sphere(const GenericObjectDevice* genObjDev,
		    const Material* mat)
    : m_bbox(AABB(genObjDev->bmin, genObjDev->bmax)),
      m_center(genObjDev->vectors[0]),
      m_radius(genObjDev->scalars[0]),
      m_material(mat) {}

  __device__ virtual bool hit(const Ray& ray, float tMin,
			      float tMax, HitRecord& hitRec) const;
  __device__ virtual bool getBBox(AABB &bbox) const {
    bbox = m_bbox;
    return true;
  }

  __device__ static void getSphereUV(const float3& p,
				     float& u, float &v);

  AABB m_bbox;
  float3 m_center;
  float m_radius;
  const Material *m_material;

private:
  __device__ float3 normalAtP(float3 point) const;
};

#endif//OBJECTS_CUH
