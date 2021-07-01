#include "AABB.cuh"

__device__
bool AABB::hit(const Ray& ray, float tMin, float tMax) const
{
  for (int a = 0; a < 3; a++) {
    float direction = ray.direction(a);
    float origin = ray.origin(a);
    float invD = 1.0f/direction;
    float t0 = (m_bmin[a] - origin)*invD;
    float t1 = (m_bmax[a] - origin)*invD;
    if (invD < 0.0f) {
      swap(t0, t1);
    }
    tMin = tMin < t0 ? t0 : tMin; 
    tMax = t1 < tMax ? t1 : tMax;
    if (tMax <= tMin) {
      return false;
    }
  }
  return true;
}
