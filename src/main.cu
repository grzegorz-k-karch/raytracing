#include "gkk_cuda_utils.cuh"

#include <iostream>

__global__ void render(float* fb, int max_x, int max_y)
{
  int i = threadIdx.x +  blockIdx.x*blockDim.x;
  int j = threadIdx.y +  blockIdx.y*blockDim.y;

  if ((i >= max_x) && (j >= max_y)) {
    return;
  }

  int pixel_idx = (i + j*max_x)*3;
  fb[pixel_idx + 0] = float(i)/max_x;
  fb[pixel_idx + 1] = float(j)/max_y;
  fb[pixel_idx + 2] = 0.2f;
}

int main()
{
  int nx = 1600;
  int ny = 800;
  int num_pixels = nx*ny;
  size_t fb_size = 3*num_pixels*sizeof(float);

  float *fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  int tx = 8;
  int ty = 8;

  dim3 blocks((nx+tx-1)/tx, (ny+ty-1)/ty);
  dim3 threads(tx, ty);

  render<<<blocks, threads>>>(fb, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_idx = (i + j*nx)*3;
      float r = fb[pixel_idx + 0];
      float g = fb[pixel_idx + 1];
      float b = fb[pixel_idx + 2];

      int ir = int(255.99f*r);
      int ig = int(255.99f*g);
      int ib = int(255.99f*b);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }


  cudaFree(fb);

  return 0;
}
