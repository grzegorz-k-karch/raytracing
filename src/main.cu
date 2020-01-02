#include "gkk_cuda_utils.cuh"
#include "gkk_vec.cuh"
#include "gkk_color.cuh"
#include "gkk_object.cuh"
#include "gkk_geometry.cuh"

#include <iostream>

#include <curand_kernel.h>

// assuming pixel values are in range (0,1)
int write_ppm(vec3* raw_image,
	      const int nx=300,
	      const int ny=200)
{
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_idx = i + j*nx;
      vec3 color = raw_image[pixel_idx];
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


__device__ bool hit_sphere(const vec3& center, float radius, const Ray& ray)
{
  vec3 oc = ray.o - center;
  float a = dot(ray.d, ray.d);
  float b = 2.0f*dot(oc, ray.d);
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - 4*a*c;

  return (discriminant > 0.0f);
}


__global__ void render(vec3* fb, int max_x, int max_y,
		       vec3 lower_left_corner, vec3 horizontal,
		       vec3 vertical, vec3 origin)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pixel_idx = i + j*max_x;
  float u = float(i)/float(max_x);
  float v = float(j)/float(max_y);
  Ray ray(origin, lower_left_corner + u*horizontal + v*vertical);
  vec3 color = get_plane_color(ray);
  vec3 sphere_center = vec3(0.0f, 0.0f, -2.0f);
  float sphere_radius = 0.5f;
  if (hit_sphere(sphere_center, sphere_radius, ray)) {
    color = vec3(1.0f, 0.0f, 0.0f);
  }
  fb[pixel_idx] = color;
}


__global__ void render(vec3* fb, int max_x, int max_y,
		       vec3 lower_left_corner, vec3 horizontal,
		       vec3 vertical, vec3 origin, Object** world)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pixel_idx = i + j*max_x;
  float u = float(i)/float(max_x);
  float v = float(j)/float(max_y);
  Ray ray(origin, lower_left_corner + u*horizontal + v*vertical);
  vec3 color = get_color(ray, *world); //, 0);
  fb[pixel_idx] = color;
}


__global__ void create_world(Object** d_list, Object** d_world)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f,
			   new Lambertian(vec3(0.5f, 0.5f, 0.5f)));
    *(d_list + 1) = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f,
			       new Lambertian(vec3(0.5f, 0.5f, 0.5f)));
    *d_world = new ObjectList(d_list, 2);
  }
}


__global__ void free_world(Object** d_list, Object** d_world)
{
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
}


void generate_test_image(vec3* raw_image,
			const int nx=400,
			const int ny=200)
{

  int tx = 8;
  int ty = 8;

  dim3 blocks((nx+tx-1)/tx, (ny+ty-1)/ty);
  dim3 threads(tx, ty);

  Object **d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(Object*)));
  Object **d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Object*)));

  create_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(raw_image, nx, ny, vec3(-2.0f, -1.0f, -1.0f),
			      vec3(4.0f, 0.0f, 0.0f), vec3(0.0, 2.0f, 0.0f),
			      vec3(0.0f, 0.0f, 0.0f), d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
}

int main()
{
  int nx = 1600;
  int ny = 800;

  int num_pixels = nx*ny;

  size_t fb_size = num_pixels*sizeof(vec3);
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  generate_test_image(fb, nx, ny);
  write_ppm(fb, nx, ny);

  cudaFree(fb);

  return 0;
}
