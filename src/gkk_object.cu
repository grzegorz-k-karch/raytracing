#include "gkk_object.cuh"


__device__
bool ObjectList::hit(const Ray& ray,
		     float t_min, float t_max,
		     hit_record& hrec) const
{
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


__device__
bool ObjectList::get_bbox(float t0, float t1, AABB& output_bbox) const
{
  if (objects == nullptr) {
    return false;
  }
  bool has_bbox = false;
  if (bbox == nullptr) {
    AABB surr_bbox;

    // get the first bounding box to initialize the  surrounding bounding box
    int i = 0;
    for (; i < num_objects; i++) {
      if (objects[0]->get_bbox(t0, t1, surr_bbox)) {
	break;
      }
    }
    // process remaining bounding boxes
    for (; i < num_objects; i++) {
      AABB tmp_bbox;
      if (objects[0]->get_bbox(t0, t1, tmp_bbox)) {
	surr_bbox = surrounding_bbox(surr_bbox, tmp_bbox);
	has_bbox = true;
      }
    }
    if (has_bbox) {
      output_bbox = surr_bbox;
    }
  }
  else {
    output_bbox = *bbox;
    has_bbox = true;
  }
  return has_bbox;
}


__device__
BVHNode::BVHNode(Object** objects, int start, int end, float time0, float time1,
		 curandState* rand_state)
{
  int axis = int(ceilf(curand_uniform(rand_state)*3.0f) - 1.0f); //  (0:1](1:2](2:3])

  int object_span = end - start;

  if (object_span == 1) {
    left = right = objects[start];
  }
  else if (object_span == 2) {
    if (compare_bboxes(objects[start], objects[start + 1], axis)) {
      left = objects[start];
      right = objects[start + 1];
    }
    else {
      left = objects[start + 1];
      right = objects[start];
    }
  }
  else {
    // TODO: sort
    int mid = start + object_span/2;
    left = new BVHNode(objects, start, mid, time0, time1, rand_state);
    right = new BVHNode(objects, mid, end, time0, time1, rand_state);
  }

  AABB box_left, box_right;
  if (!left->get_bbox(time0, time1, box_left) || !right->get_bbox(time0, time1, box_right))  {
    printf("|||| No bounding box in BVHNode constructor.\n");
  }

  bbox = new AABB(surrounding_bbox(box_left, box_right));
}


__device__
bool BVHNode::hit(const Ray& ray, float t_min, float t_max, hit_record& hrec) const
{
  if (!bbox->hit(ray, t_min, t_max)) {
    return false;
  }

  bool hit_left = left->hit(ray, t_min, t_max, hrec);
  bool hit_right = right->hit(ray, t_min, hit_left ? hrec.t : t_max, hrec);

  return hit_left || hit_right;
}


__device__
bool BVHNode::get_bbox(float t0, float t1, AABB& output_bbox) const
{
  if (bbox == nullptr) {
    return false;
  }
  output_bbox = *bbox;
  return true;
}
