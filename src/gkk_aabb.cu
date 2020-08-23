#include "gkk_aabb.cuh"

std::ostream& operator<<(std::ostream& os, const AABB& bbox)
{
  return os << "(" << bbox.bmin << ", " << bbox.bmax << ")";
}
