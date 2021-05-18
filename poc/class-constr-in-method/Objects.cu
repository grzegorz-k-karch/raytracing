#include "Objects.cuh"

__device__
BVHNode::BVHNode(int depth) :
  m_depth(depth) {
  if (depth > 0) {
    m_left = new BVHNode(depth-1);
    m_right = new BVHNode(depth-1);
#if __CUDA_ARCH__ >= 200
    printf("|||| depth %d left %p right %p\n", m_depth,m_left,m_right);
#endif
  } else {
    m_left = nullptr;
    m_right = nullptr;
#if __CUDA_ARCH__ >= 200
    printf("|||| depth %d left %p right %p\n", m_depth,m_left,m_right);
#endif
      
  }
}
