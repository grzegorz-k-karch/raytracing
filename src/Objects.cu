#include "Objects.cuh"
#include "nvidia/helper_math.h"
#include "assert.h"

__device__
bool ObjectList::hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const
{
  bool hitAny = false;
  HitRecord closestHitRec;
  float closestSoFar = tMax;
  for (int i = 0; i < num_objects; i++) {
    if (objects[i]->hit(ray, tMin, closestSoFar, closestHitRec)) {
      hitAny = true;
      closestSoFar = closestHitRec.t;
      hitRec = closestHitRec;
    }
  }
  return hitAny;
}


__device__
bool ObjectList::getBBox(AABB& outBBox) const
{
  if (objects == nullptr) {
    return false;
  }
  bool hasBBox = false;
  if (!bboxComputed) {
    AABB objBBox;
    bool firstBBox = true;
    for (int i = 0; i < num_objects; i++) {
      if (objects[i]->getBBox(objBBox)) {
  	outBBox = firstBBox ? objBBox : surroundingBBox(outBBox, objBBox);
        firstBBox = false;
  	hasBBox = true;
      }
    }
  }
  return hasBBox;
}


// __device__
// void mergeObjects(Object** objects, int start, int mid, int end, int axis)
// {
//   int numLeft = mid - start + 1;
//   int numRight = end - mid;

//   Object **objectsLeft = new Object*[numLeft];
//   Object **objectsRight = new Object*[numRight];

//   for (int i = 0; i < numLeft; i++) {
//     objectsLeft[i] = objects[start+i];
//   }
//   for (int i = 0; i < numRight; i++) {
//     objectsRight[i] = objects[mid+1+i];
//   }

//   int leftIdx = 0;
//   int rightIdx = 0;

//   int resultIdx = start;

//   while (leftIdx < numLeft && rightIdx < numRight) {
//     if (compareBBoxes(objectsLeft[leftIdx], objectsRight[rightIdx], axis)) {
//       objects[resultIdx] = objectsLeft[leftIdx];
//       leftIdx++;
//     }
//     else {
//       objects[resultIdx] = objectsRight[rightIdx];
//       rightIdx++;
//     }
//     resultIdx++;
//   }

//   delete [] objectsLeft;
//   delete [] objectsRight;
// }

// __device__
// void sortObjects(Object** objects, int start, int end, int axis)
// {
//   if (start < end) {
//     int mid = start + (end - start)/2;
//     sortObjects(objects, start, mid, axis);
//     sortObjects(objects, mid+1, end, axis);
//     mergeObjects(objects, start, mid, end, axis);
//   }
// }


// __device__
// BVHNode::BVHNode(Object** objects, int start, int end, curandState* randState)
// {
//   int axis = int(ceilf(curand_uniform(randState)*3.0f) - 1.0f); //  (0:1](1:2](2:3])
//   int objectSpan = end - start;

//   if (objectSpan == 1) {
//     m_left = m_right = objects[start];
//   }
//   else if (objectSpan == 2) {
//     if (compareBBoxes(objects[start], objects[start + 1], axis)) {
//       m_left = objects[start];
//       m_right = objects[start + 1];
//     }
//     else {
//       m_left = objects[start + 1];
//       m_right = objects[start];
//     }
//   }
//   else {
//     sortObjects(objects, start, end-1, axis);
//     int mid = start + objectSpan/2;
//     m_left = new BVHNode(objects, start, mid, randState);
//     m_right = new BVHNode(objects, mid, end, randState);
//   }

//   AABB boxLeft, boxRight;
//   if (!m_left->getBBox(boxLeft) || !m_right->getBBox(boxRight))  {
// #if __CUDA_ARCH__ >= 200
//     printf("|||| No bounding box in BVHNode constructor.\n");
// #endif//__CUDA_ARCH__ >= 200
//   }

//   m_bbox = AABB(surroundingBBox(boxLeft, boxRight));
//   m_bboxComputed = true;
// }


__device__
bool BVHNode::hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const
{
  if (!(m_bbox.hit(ray, tMin, tMax))) {
    return false;
  }

  bool hitLeft = m_left->hit(ray, tMin, tMax, hitRec);
  bool hitRight = m_right->hit(ray, tMin, hitLeft ? hitRec.t : tMax, hitRec);

  return hitLeft || hitRight;
}


__device__
bool BVHNode::getBBox(AABB& outBBox) const
{
  if (m_bboxComputed) {
    outBBox = m_bbox;
    return true;
  }
  return false;
}

__device__
Object* createBVH(Object **objects, int numObjects)
{
  return new ObjectList(objects, numObjects);
}


__device__
float3 Mesh::normalAtP(float u, float v,
		       float3 n0, float3 n1, float3 n2) const
{
  float3 n = (1.0f - u - v)*n0 + u*n1 + v*n2;
  n = normalize(n);

  return n;
}


// from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
// code rewritten to do tests on the sign of the determinant
// the division is before the test of the sign of the det
// based on variant intersect_triangle2
__device__ int intersectTriangle(float3 orig, float3 dir,
				 float3 vert0, float3 vert1, float3 vert2,
				 float* t, float* u, float* v)
{
  const float EPSILON = 0.0001f;
  // find vectors for two edges sharing vert0
  float3 edge1 = vert1 - vert0;
  float3 edge2 = vert2 - vert0;

  // begin calculating determinant - also used to calculate U parameter
  float3 pvec = cross(dir, edge2);

  // if determinant is near zero, ray lies in plane of triangle
  float det = dot(edge1, pvec);

  // calculate distance from vert0 to ray origin
  float3 tvec = orig - vert0;
  float inv_det = 1.0f/det;
  float3 qvec;
  if (det > EPSILON) {
    // calculate U parameter and test bounds
    *u = dot(tvec, pvec);
    if (*u < 0.0 || *u > det) {
      return 0;
    }
    // prepare to test V parameter
    qvec = cross(tvec, edge1);

    // calculate V parameter and test bounds
    *v = dot(dir, qvec);
    if (*v < 0.0 || *u + *v > det) {
      return 0;
    }
  }
  else if(det < -EPSILON) {
    // calculate U parameter and test bounds
    *u = dot(tvec, pvec);
    if (*u > 0.0 || *u < det) {
      return 0;
    }
    // prepare to test V parameter
    qvec = cross(tvec, edge1);

    // calculate V parameter and test bounds
    *v = dot(dir, qvec) ;
    if (*v > 0.0 || *u + *v < det)
      return 0;
  }
  else {
    return 0;  // ray is parallell to the plane of the triangle
  }

  // calculate t, ray intersects triangle
  *t = dot(edge2, qvec)*inv_det;
  (*u) *= inv_det;
  (*v) *= inv_det;

  return 1;
}


__device__
bool Mesh::hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const
{
  if (!(m_bbox.hit(ray, tMin, tMax))) {
    return false;
  }
  float u, v;
  float t = 3.402823e+38;
  // int isect = 0;
  int tidx;

  int numTriangles = numTriangleIndices/3;
  // ensure numTriangleIndices is divisible by 3
  assert(numTriangles*3 == numTriangleIndices);

  for (int triangleIdx = 0; triangleIdx < numTriangles; triangleIdx++) {

    int v0 = triangleIndices[triangleIdx*3];
    int v1 = triangleIndices[triangleIdx*3 + 1];
    int v2 = triangleIndices[triangleIdx*3 + 2];

    float3 vert0 = vertices[v0];
    float3 vert1 = vertices[v1];
    float3 vert2 = vertices[v2];

    float t_tmp, u_tmp, v_tmp;
    int isect = intersectTriangle(ray.m_origin, ray.m_direction,
				  vert0, vert1, vert2,
				  &t_tmp, &u_tmp, &v_tmp);
    if (isect) {
      if (t_tmp < t) {
	t = t_tmp;
	u = u_tmp;
	v = v_tmp;
	tidx = triangleIdx;
      }
    }
  }

  if (t > tMin && t < tMax) {
    hitRec.t = t;
    hitRec.p = ray.pointAtT(t);

    int v0 = triangleIndices[tidx*3];
    int v1 = triangleIndices[tidx*3 + 1];
    int v2 = triangleIndices[tidx*3 + 2];
    float3 n0 = vertexNormals[v0];
    float3 n1 = vertexNormals[v1];
    float3 n2 = vertexNormals[v2];
    hitRec.n = normalAtP(u, v, n0, n1, n2);

    hitRec.material = m_material;
    return true;
  }
  return false;
}


__device__
float3 Sphere::normalAtP(float3 point) const
{
  return normalize(point - m_center);
}


__device__
bool Sphere::hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const
{
  if (!(m_bbox.hit(ray, tMin, tMax))) {
    return false;
  }
  float3 oc = ray.m_origin - m_center;
  float3 d = ray.m_direction;
  // computing discriminant for ray-sphere intersection
  float a = dot(d, d);
  float b = 2.0f*dot(d, oc);
  float c = dot(oc, oc) - m_radius*m_radius;
  float discriminant = b*b - 4.0f*a*c;
  float t = -1.0f;
  if (discriminant > 0.0f) {
    float x1 = (-b - sqrtf(discriminant))/(2.0f*a);
    float x2 = (-b + sqrtf(discriminant))/(2.0f*a);
    t = fminf(x1, x2);
    if (t > tMin && t < tMax) {
      hitRec.t = t;
      hitRec.p = ray.pointAtT(t);
      hitRec.n = normalAtP(hitRec.p);
      hitRec.material = m_material;
      return true;
    }
  }
  return false;
}


// __device__
// Object* createBVH(Object **objects, int numObjects)
// {
//   curandState localRandState;
//   curand_init(1984, 0, 0, &localRandState);

//   int numLeavesPerSubtree = numObjects;
//   // first, sort objects in divide and conquer manner,
//   // change sorting axis in each "devide"
//   while (numLeavesPerSubtree > 0) {

//     int axis = int(ceilf(curand_uniform(&localRandState)*3.0f) - 1.0f);

//     for (int start = 0; start < numObjects; start+=numLeavesPerSubtree) {

//       int objectSpan = start + numLeavesPerSubtree <= numObjects ?
//   	numLeavesPerSubtree : numObjects - start;

//       if (objectSpan == 2) {
// #if __CUDA_ARCH__ >= 200
// 	printf("|||| %p %p %d %d\n", objects[start+1], objects[start], axis, start);
// #endif
//       	if (compareBBoxes(objects[start + 1], objects[start], axis)) {
//       	  // left objects[start] is larger than right objects[start+1]
//       	  // so we need to swap them
//       	  // Object *tmp = objects[start];
//       	  // objects[start] = objects[start+1];
//       	  // objects[start+1] = tmp;
//       	}
//       }
//       // else {
//       // 	// sortObjects(objects, start, start+objectSpan-1, axis);
//       // }
//     }
//     numLeavesPerSubtree /= 2;
//   }
//   return new ObjectList(objects, numObjects);
// }

