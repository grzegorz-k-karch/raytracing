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
  bool getBBox(AABB& bbox) const = 0;  
};


class ObjectList: public Object {
 public:
  __device__ ObjectList(Object** objects, int num_objects) :
    objects(objects), num_objects(num_objects), bboxComputed(false) {}

  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(AABB& bbox) const;

  Object **objects;
  int num_objects;

  AABB bbox;
  bool bboxComputed;
};


class BVHNode : public Object {
public:
  __device__ BVHNode(Object** objects, int start, int end,
  		     curandState* randState);

  __device__ BVHNode() :
    m_left(nullptr), m_right(nullptr), m_bboxComputed(false) {}

  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(AABB& outBBox) const;

  // __device__
  // void setChildren(BVHNode* left, BVHNode* right) {
  //   m_left = left;
  //   m_right = right;

  //   if (m_left) {
  //     AABB boxLeft;
  //     if (m_left->getBBox(boxLeft)) {
  // 	m_bbox = boxLeft;
  // 	m_bboxComputed = true;
  //     }
  //   }

  //   if (m_right) {
  //     AABB boxRight;
  //     if (m_left->getBBox(boxRight)) {
  //       if (m_bboxComputed) {
  // 	  m_bbox = AABB(surroundingBBox(m_bbox, boxRight));
  //       }
  // 	else {
  // 	  m_bbox = boxRight;
  // 	  m_bboxComputed = true;
  //       }
  //     }
  //   }    
  // }

public:
  Object *m_left;
  Object *m_right;

  AABB m_bbox;
  bool m_bboxComputed;  
};


__device__
Object* createBVH(Object** objects, int numObjects);


__device__ __inline__
bool compareBBoxes(Object* a, Object* b, int axis)
{
  AABB bboxA;
  AABB bboxB;
  
  // if (!(a->getBBox(bboxA)) ||
  //     !(b->getBBox(bboxB))) {
  //   return false;
  // }
  // float3 Amin = bboxA.min();
  // float3 Bmin = bboxB.min();
  // if (axis == 0) {
  //   return Amin.x < Bmin.x;
  // }
  // else if (axis == 1) {
  //   return Amin.y < Bmin.y;
  // }
  // else {
  //   return Amin.z < Bmin.z;    
  // }
  return false;
}


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
  bool getBBox(AABB &bbox) const {
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
  float3 normalAtP(float u, float v,
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
  
  __device__ virtual
  bool hit(const Ray& ray, float tMin,
	   float tMax, HitRecord& hitRec) const;
  __device__ virtual
  bool getBBox(AABB &bbox) const {
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
