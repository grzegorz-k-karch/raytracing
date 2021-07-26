#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include "Ray.cuh"
#include "GenericObject.h"
#include "Materials.cuh"


class Object {
public:
  __device__ virtual ~Object() {}
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
  __device__ ~BVHNode() {
    if (m_left) {
#if __CUDA_ARCH__ >= 200
      printf("delete left\n");
#endif
      delete m_left;
    }
    if (m_right) {
#if __CUDA_ARCH__ >= 200
      printf("delete right\n");
#endif
      delete m_right;
    }
  }

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
__device__ void destroyBVH(BVHNode** root);

__device__ bool compareBBoxes(Object* a, Object* b, int axis);


class Mesh : public Object {
public:
  __device__ Mesh(const GenericObjectDevice* genObjDev)
    : m_bbox(AABB(genObjDev->m_bmin, genObjDev->m_bmax)),
      m_material(MaterialFactory::createMaterial(genObjDev->m_material)),
      vertices(genObjDev->m_vertices),
      numVertices(genObjDev->m_numVertices),
      vertexColors(genObjDev->m_vertexColors),
      numVertexColors(genObjDev->m_numVertexColors),
      vertexNormals(genObjDev->m_vertexNormals),
      numVertexNormals(genObjDev->m_numVertexNormals),
      textureCoords(genObjDev->m_textureCoords),
      numTextureCoords(genObjDev->m_numTextureCoords),
      triangleIndices(genObjDev->m_triangleIndices),
      numTriangleIndices(genObjDev->m_numTriangleIndices),
      m_smoothness(genObjDev->m_scalars[0]) {}

  __device__ ~Mesh() {
#if __CUDA_ARCH__ >= 200
    printf("~Mesh\n");
#endif
    if (m_material) {
      delete m_material;
    }
  }

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
  __device__ Sphere(const GenericObjectDevice* genObjDev)
    : m_bbox(AABB(genObjDev->m_bmin, genObjDev->m_bmax)),
      m_material(MaterialFactory::createMaterial(genObjDev->m_material)),
      m_center(genObjDev->m_vectors[0]),
      m_radius(genObjDev->m_scalars[0]) {}

  __device__ ~Sphere() {
#if __CUDA_ARCH__ >= 200
    printf("~Sphere\n");
#endif
    if (m_material) {
      delete m_material;
    }
  }

  __device__ virtual bool hit(const Ray& ray, float tMin,
			      float tMax, HitRecord& hitRec) const;
  __device__ virtual bool getBBox(AABB &bbox) const {
    bbox = m_bbox;
    return true;
  }
  __device__ static void getSphereUV(const float3& p, float& u, float &v);

  AABB m_bbox;
  float3 m_center;
  float m_radius;
  const Material *m_material;

private:
  __device__ float3 normalAtP(float3 point) const;
};

#endif//OBJECTS_CUH
