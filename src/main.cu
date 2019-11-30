#include "gkk_cuda_utils.cuh"
#include "gkk_vec.cuh"
#include "gkk_color.cuh"

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

// assuming pixel values are in range (0,1)
int write_ppm(const float* raw_image,
	      const int nx=300,
	      const int ny=200)
{
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_idx = (i + j*nx)*3;
      vec3 color = vec3(raw_image[pixel_idx + 0],
			raw_image[pixel_idx + 1],
			raw_image[pixel_idx + 2]);
      // gamma correction
      color = vec3(std::sqrt(color.r()), std::sqrt(color.g()), std::sqrt(color.b()));
      int ir = int(255.99f*color.r());
      int ig = int(255.99f*color.g());
      int ib = int(255.99f*color.b());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  return 0;
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

  write_ppm(fb, nx, ny);

  cudaFree(fb);

  return 0;
}
