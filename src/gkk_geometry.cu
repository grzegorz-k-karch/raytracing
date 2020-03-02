#include "gkk_geometry.cuh"
#include "gkk_material.cuh"

#include <algorithm>

__device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const {

  vec3 oc = ray.origin() - center;
  vec3 d = ray.direction();
  // computing discriminant for ray-sphere intersection
  float a = dot(d, d);
  float b = 2.0f*dot(d, oc);
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - 4.0f*a*c;
  float t = -1.0f;
  if (discriminant > 0.0f) {
    float x1 = (-b - sqrtf(discriminant))/(2.0f*a);
    float x2 = (-b + sqrtf(discriminant))/(2.0f*a);
    t = fminf(x1, x2);
    if (t > t_min && t < t_max) {
      hrec.t = t;
      hrec.p = ray.point_at_t(t);
      hrec.n = normal_at_p(hrec.p);
      hrec.material_ptr = material_ptr;
      return true;
    }
  }
  return false;
}

__device__ vec3 Sphere::normal_at_p(const vec3& point) const
{
  return normalize(point - center);
}


// from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
// code rewritten to do tests on the sign of the determinant
// the division is before the test of the sign of the det
// based on variant intersect_triangle2
__device__
int intersect_triangle(vec3 orig, vec3 dir,
		       vec3 vert0, vec3 vert1, vec3 vert2,
		       float* t, float* u, float* v)
{
  const float EPSILON = 0.0001f;
  // find vectors for two edges sharing vert0
  vec3 edge1 = vert1 - vert0;
  vec3 edge2 = vert2 - vert0;

  // begin calculating determinant - also used to calculate U parameter
  vec3 pvec = cross(dir, edge2);

  // if determinant is near zero, ray lies in plane of triangle
  float det = dot(edge1, pvec);

  // calculate distance from vert0 to ray origin
  vec3 tvec = orig - vert0;
  float inv_det = 1.0f/det;
  vec3 qvec;
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
vec3 TriangleMesh::normal_at_p(const vec3& point,
			       const vec3 vert0,
			       const vec3 vert1,
			       const vec3 vert2) const
{
  vec3 e0 = vert1 - vert0;
  vec3 e1 = vert2 - vert0;
  vec3 n = cross(e0, e1);
  n = normalize(n);
  
  return n;
}


__device__
bool TriangleMesh::hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const
{
  float u, v;
  float t = 3.402823e+38;
  int isect = 0;
  int tidx;
  for (int i = 0; i < num_triangles; i++) {

    int v0 = triangle_list[i*3];
    int v1 = triangle_list[i*3 + 1];
    int v2 = triangle_list[i*3 + 2];
    
    vec3 vert0 = point_list[v0];
    vec3 vert1 = point_list[v1];
    vec3 vert2 = point_list[v2];

    float t_tmp, u_tmp, v_tmp;
    int isect = intersect_triangle(ray.origin(), ray.direction(),
				   vert0, vert1, vert2,
				   &t_tmp, &u_tmp, &v_tmp);
    if (isect) {
      if (t_tmp < t) {
	t = t_tmp;
	u = u_tmp;
	v = v_tmp;
	tidx = i;
      }
    }
  }

  if (t > t_min && t < t_max) {
    hrec.t = t;
    int v0 = triangle_list[tidx*3];
    int v1 = triangle_list[tidx*3 + 1];
    int v2 = triangle_list[tidx*3 + 2];
    vec3 vert0 = point_list[v0];
    vec3 vert1 = point_list[v1];
    vec3 vert2 = point_list[v2];

    hrec.p = ray.point_at_t(t);
    hrec.n = normal_at_p(hrec.p, vert0, vert1, vert2);
    hrec.material_ptr = material_ptr;
    return true;
  }
  return false;
}
