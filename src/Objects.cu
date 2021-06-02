#include "Objects.cuh"
#include "nvidia/helper_math.h"
#include "assert.h"


__device__
bool compareBBoxes(Object* a, Object* b, int axis)
{
  AABB bboxA;
  AABB bboxB;

  if (!(a->getBBox(bboxA)) ||
      !(b->getBBox(bboxB))) {
    return false;
  }
  float3 Amin = bboxA.min();
  float3 Bmin = bboxB.min();
  if (axis == 0) {
    return Amin.x < Bmin.x;
  }
  else if (axis == 1) {
    return Amin.y < Bmin.y;
  }
  else {
    return Amin.z < Bmin.z;
  }
}


__device__
void sortObjects(Object **objects, int numObjects, int axis)
{
  Object **sortedObjects = new Object*[numObjects];

  int stride = 1;
  while (stride < numObjects) {
    for (int offset = 0; offset < numObjects; offset += stride*2) {
      int p = offset;
      int q = p + stride;
      int r = q + stride < numObjects ? q + stride : numObjects;
      int i = p;
      int j = q;
      for (int k = p; k < r; k++) {
        if (i < q && j < r) {
	  if (compareBBoxes(objects[i], objects[j], axis)) {
	    sortedObjects[k] = objects[i];
	    i++;
	  }
	  else {
	    sortedObjects[k] = objects[j];
	    j++;
	  }
        }
	else {
          if (i < q) {
	    sortedObjects[k] = objects[i];
	    i++;
          } else {
	    sortedObjects[k] = objects[j];
	    j++;
          }
        }
      }
    }
    for (int objIdx = 0; objIdx < numObjects; objIdx++) {
      objects[objIdx] = sortedObjects[objIdx];
    }
    stride *= 2;
  }
  delete [] sortedObjects;
}


__device__
void sortNodes(BVHNode **nodes, int numNodes, int axis)
{
  BVHNode **sortedNodes = new BVHNode*[numNodes];

  int stride = 1;
  while (stride < numNodes) {
    for (int offset = 0; offset < numNodes; offset += stride*2) {
      int p = offset;
      int q = p + stride;
      int r = q + stride < numNodes ? q + stride : numNodes;
      int i = p;
      int j = q;
      for (int k = p; k < r; k++) {
        if (i < q && j < r) {
	  if (compareBBoxes(nodes[i], nodes[j], axis)) {
	    sortedNodes[k] = nodes[i];
	    i++;
	  }
	  else {
	    sortedNodes[k] = nodes[j];
	    j++;
	  }
        }
	else {
          if (i < q) {
	    sortedNodes[k] = nodes[i];
	    i++;
          } else {
	    sortedNodes[k] = nodes[j];
	    j++;
          }
        }
      }
    }
    for (int objIdx = 0; objIdx < numNodes; objIdx++) {
      nodes[objIdx] = sortedNodes[objIdx];
    }
    stride *= 2;
  }
  delete [] sortedNodes;
}


__device__
Object* createBVH(Object **objects, int numObjects)
{
  BVHNode *root = nullptr;
  curandState localRandState;
  curand_init(1984, 0, 0, &localRandState);

  int numNodes = (numObjects+1)/2;  // number of BVH leaf nodes (=numObjects/2)
  BVHNode **nodes = new BVHNode*[numNodes];

  int axis = int(ceilf(curand_uniform(&localRandState)*3.0f) - 1.0f);
  sortObjects(objects, numObjects, axis);
  for (int pairIdx = 0; pairIdx < numNodes; pairIdx++) {
    Object *left = objects[pairIdx*2];
    int rightIdx = pairIdx*2 + 1;
    Object *right = rightIdx < numObjects ? objects[rightIdx] : nullptr;
    nodes[pairIdx] = new BVHNode(left, right);
  }

  while (0 < numNodes/2) {
    axis = int(ceilf(curand_uniform(&localRandState)*3.0f) - 1.0f);
    sortNodes(nodes, numNodes, axis);
    for (int pairIdx = 0; pairIdx < numNodes; pairIdx++) {
      Object *left = nodes[pairIdx*2];
      int rightIdx = pairIdx*2 + 1;
      Object *right = rightIdx < numNodes ? nodes[rightIdx] : nullptr;
      nodes[pairIdx] = new BVHNode(left, right);
    }
    numNodes /= 2;
  }

  root = nodes[0];

  delete [] nodes;

  return root;
}


__device__
bool BVHNode::hit(const Ray& ray, float tMin, float tMax, HitRecord& hitRec) const
{
  if (!(m_bbox.hit(ray, tMin, tMax))) {
    return false;
  }

  bool hitLeft = m_left != nullptr && m_left->hit(ray, tMin, tMax, hitRec);
  bool hitRight = m_right != nullptr && m_right->hit(ray, tMin, hitLeft ? hitRec.t : tMax, hitRec);

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
void BVHNode::setChildren(Object* left, Object* right)
{
  m_left = left;
  m_right = right;

  if (m_left) {
    AABB boxLeft;
    if (m_left->getBBox(boxLeft)) {
      m_bbox = boxLeft;
      m_bboxComputed = true;
    }
  }

  if (m_right) {
    AABB boxRight;
    if (m_right->getBBox(boxRight)) {
      if (m_bboxComputed) {
	m_bbox = AABB(surroundingBBox(m_bbox, boxRight));
      }
      else {
	m_bbox = boxRight;
	m_bboxComputed = true;
      }
    }
  }
}


__device__ float3 Mesh::normalAtP(float u, float v,
				  float3 n0, float3 n1, float3 n2) const
{
  float3 n =
    powf(1.0f - u - v, m_smoothness)*n0 +
    powf(u, m_smoothness)*n1 +
    powf(v, m_smoothness)*n2;
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

    int t0 = triangleIndices[tidx*3];
    int t1 = triangleIndices[tidx*3 + 1];
    int t2 = triangleIndices[tidx*3 + 2];
    float3 n0 = vertexNormals[t0];
    float3 n1 = vertexNormals[t1];
    float3 n2 = vertexNormals[t2];
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
      getSphereUV(hitRec.n, hitRec.u, hitRec.v);
      hitRec.material = m_material;
      return true;
    }
  }
  return false;
}

__device__ void Sphere::getSphereUV(const float3& p,
				    float& u, float &v)
{
  const float pi = 3.14159265f;
  float theta = acosf(p.y);
  float phi = atan2f(-p.z, p.x) + pi;
  u = phi/(2*pi);
  v = theta/pi;
}

