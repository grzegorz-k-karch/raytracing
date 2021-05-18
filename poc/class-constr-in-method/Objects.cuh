#ifndef OBJECTS_CUH
#define OBJECTS_CUH

class BVHNode {
public:
  __device__ BVHNode(int depth);
  BVHNode *m_left;
  BVHNode *m_right;
  int m_depth;
};

#endif//OBJECTS_CUH
