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
float3 Mesh::normalAtP(float3 point,
		       const float3 vert0,
		       const float3 vert1,
		       const float3 vert2) const
{
  float3 e0 = vert1 - vert0;
  float3 e1 = vert2 - vert0;
  float3 n = cross(e0, e1);
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
  // float u, v;
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
	tidx = triangleIdx;
      }
    }
  }

  if (t > tMin && t < tMax) {
    hitRec.t = t;
    int v0 = triangleIndices[tidx*3];
    int v1 = triangleIndices[tidx*3 + 1];
    int v2 = triangleIndices[tidx*3 + 2];
    float3 vert0 = vertices[v0];
    float3 vert1 = vertices[v1];
    float3 vert2 = vertices[v2];

    hitRec.p = ray.pointAtT(t);
    hitRec.n = normalAtP(hitRec.p, vert0, vert1, vert2);
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
