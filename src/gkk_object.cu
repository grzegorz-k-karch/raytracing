#include "gkk_object.cuh"

__host__ __device__ bool ObjectList::hit(const Ray& ray,
					 float t_min, float t_max,
					 hit_record& hrec) const {
  
  bool hit_any = false;
  hit_record tmp_hrec;
  float closest_so_far = t_max;
  for (int i = 0; i < num_objects; i++) {
    if (objects[i]->hit(ray, t_min, closest_so_far, tmp_hrec)) {
      hit_any = true;
      closest_so_far = tmp_hrec.t;
      hrec = tmp_hrec;
    }
  }
  return hit_any;
}
