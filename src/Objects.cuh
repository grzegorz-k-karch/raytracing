#ifndef OBJECTS_CUH
#define OBJECTS_CUH

class Material;


struct HitRecord {
  float t; // time
  float3 p; // position
  float3 n; // normal at p
  Material *material;
};


class Object {
 public:
  __device__ virtual void print() const = 0;
};


class ObjectList: public Object {
 public:
  __device__ ObjectList(Object** objects, int num_objects) :
    objects(objects), num_objects(num_objects) {}

//   __device__ virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const;
//   __device__ virtual bool get_bbox(float t0, float t1, AABB& output_bbox) const;

  __device__ virtual void print() const {};

  Object **objects;
  int num_objects;

//   AABB *bbox;
};


// class BVHNode : public Object {
// public:
//   __device__ BVHNode() {}
//   __device__ BVHNode(ObjectList& object_list, float time0, float time1, curandState* rand_state) :
//     BVHNode(object_list.objects, 0, object_list.num_objects, time0, time1, rand_state) {}
//   __device__ BVHNode(Object** objects, int start, int end, float time0, float time1,
//   		     curandState* rand_state);

//   __device__ virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const;
//   __device__ virtual bool get_bbox(float t0, float t1, AABB& output_bbox) const;

// public:
//   Object *left;
//   Object *right;

//   AABB *bbox;
// };


// __device__ inline bool compare_bboxes(Object* a, Object* b, int axis)
// {
//   AABB bbox_a;
//   AABB bbox_b;

//   if (!a->get_bbox(0.0f, 0.0f, bbox_a) || !b->get_bbox(0.0f, 0.0f, bbox_b)) {
//     return false;
//   }
//   return bbox_a.min().e[axis] <  bbox_b.min().e[axis];
// }


class Mesh : public Object {
public:
  __device__ Mesh(const GenericObjectDevice* genObjDev,
		  const Material* mat)
    : vertices(genObjDev->vertices),
      numVertices(genObjDev->numVertices),
      vertexColors(genObjDev->vertexColors),
      numVertexColors(genObjDev->numVertexColors),
      vertexNormals(genObjDev->vertexNormals),
      numVertexNormals(genObjDev->numVertexNormals),
      triangleIndices(genObjDev->triangleIndices),
      numTriangleIndices(genObjDev->numTriangleIndices) {}

  __device__ virtual void print() const {
  };

  float3 *vertices;
  int    numVertices;
  float3 *vertexColors;
  int    numVertexColors;
  float3 *vertexNormals;
  int    numVertexNormals;
  int *triangleIndices;
  int numTriangleIndices;
};


class Sphere : public Object {
public:
  __device__ Sphere(const GenericObjectDevice* genObjDev,
		    const Material* mat)
    : center(genObjDev->vectors[0]),
      radius(genObjDev->scalars[0]) {}
  
  __device__ virtual void print() const {
#if __CUDA_ARCH__ >= 200    
    printf("|||| radius = %f\n", radius);
#endif
  };
  
  float3 center;
  float radius;
};


#endif//OBJECTS_CUH
