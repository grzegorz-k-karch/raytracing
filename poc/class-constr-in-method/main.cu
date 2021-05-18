#include "Objects.cuh"

__global__
void constructScene_kernel(int depth, BVHNode** root)
{
  if (depth > 2) {
    *root = new BVHNode(depth);
  }
}


int main()
{
  BVHNode **root;
  cudaMalloc((void**)&root, sizeof(BVHNode*));
  constructScene_kernel<<<1,1>>>(3, root);
  cudaFree(root);
  return 0;
}
