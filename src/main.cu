#include "gkk_cuda_utils.cuh"
#include "gkk_vec.cuh"
#include "gkk_color.cuh"
#include "gkk_object.cuh"
#include "gkk_geometry.cuh"
#include "gkk_camera.cuh"

#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int load_obj(std::string input,
	     std::vector<vec3>& point_list,
	     std::vector<int>& triangle_list)
{
  std::fstream fs(input, std::fstream::in);
  char c;
  float x, y, z;
  int prev_pos = 0;
  while (fs >> c >> x >> y >> z) {
    if (c != 'v') {
      fs.seekg(prev_pos, fs.beg);
      break;
    }
    prev_pos = fs.tellg();
    vec3 v(x, y, z);
    point_list.push_back(v);
  }

  int i0, i1, i2;
  while (fs >> c >> i0 >> i1 >> i2) {
    if (c != 'f') {
      break;
    }
    triangle_list.push_back(i0-1);
    triangle_list.push_back(i1-1);
    triangle_list.push_back(i2-1);
  }

  fs.close();
  return 0;
}

// assuming pixel values are in range (0,1)
int write_ppm(vec3* raw_image,
	      const int nx,
	      const int ny,
	      std::string output)
{
  std::fstream fs(output, std::fstream::out);
  fs << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_idx = i + j*nx;
      vec3 color = raw_image[pixel_idx];
      // gamma correction
      color = vec3(std::sqrt(color.r()), std::sqrt(color.g()), std::sqrt(color.b()));
      int ir = int(255.99f*color.r());
      int ig = int(255.99f*color.g());
      int ib = int(255.99f*color.b());
      fs << ir << " " << ig << " " << ib << "\n";
    }
  }
  fs.close();
  return 0;
}


__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pixel_idx = i + j*max_x;
  curand_init(1984, pixel_idx, 0, &rand_state[pixel_idx]);
}


__global__ void render(vec3* fb, int max_x, int max_y, int ns,
		       Object** world, curandState* rand_state)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pixel_idx = i + j*max_x;
  curandState local_rand_state = rand_state[pixel_idx];
  vec3 color = vec3(0.0f, 0.0f, 0.0f);

  // default camera
  vec3 lookfrom = vec3(-1.0f, 2.7f, 4.5f);
  vec3 lookat = vec3(0.0f, 1.0f, 0.0f);
  vec3 up = vec3(0.0f, 1.0f, 0.0f);
  Camera camera(lookfrom, lookat, up, 60.0f, float(max_x)/float(max_y),
		0.125f, (lookfrom-lookat).length());

  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state))/float(max_x);
    float v = float(j + curand_uniform(&local_rand_state))/float(max_y);
    Ray ray = camera.get_ray(u, v, &local_rand_state);
    color += get_color(ray, *world, &local_rand_state);
  }
  fb[pixel_idx] = color/float(ns);
}


__global__ void create_world(Object** obj_list, Object** world, int n,
			     curandState* rand_state,
			     vec3* point_list, int num_points,
			     int* triangle_list, int num_triangles)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = rand_state[0];

    obj_list[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
			     new Lambertian(vec3(0.5f, 0.5f, 0.5f)));

    obj_list[1] = new Sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
    obj_list[2] = new Sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(vec3(0.4f, 0.2f, 0.1f)));
    obj_list[3] = new Sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    obj_list[4] = new TriangleMesh(point_list, num_points, triangle_list, num_triangles,
				   new Metal(vec3(1.0f, 0.6f, 0.5f), 0.0f));

    int i = 5;
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
    	float choose_mat = curand_uniform(&local_rand_state);
    	vec3 center(a + 0.9f*curand_uniform(&local_rand_state),
    		    0.2f,
    		    b + 0.9f*curand_uniform(&local_rand_state));
    	if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
    	  if (choose_mat < 0.5f) { // diffuse
    	    obj_list[i++] =
    	      new Sphere(center, 0.2f,
    			 new Lambertian(vec3(curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state),
    					     curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state),
    					     curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state))));
    	  }
    	  else if (choose_mat < 0.75f) { // metal
    	    obj_list[i++] =
    	      new Sphere(center, 0.2f,
    			 new Metal(vec3(0.5f*(1.0f + curand_uniform(&local_rand_state)),
    					0.5f*(1.0f + curand_uniform(&local_rand_state)),
    					0.5f*(1.0f + curand_uniform(&local_rand_state))),
    				   0.5f*curand_uniform(&local_rand_state)));
    	  }
    	  else { // glass
    	    obj_list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
    	  }
    	}
    	if (i >= n) {
    	  break;
    	}
      }
      if (i >= n) {
    	break;
      }
    }
    *world = new ObjectList(obj_list, n);
  }
}


__global__ void free_world(Object** d_list, Object** d_world, int n)
{
  for (int i = 0; i < n; i++) {
    delete d_list[i];
  }
  delete *d_world;
}


void generate_test_image(vec3* raw_image,
			 const int nx, const int ny,
			 const int num_samples)
{

  int tx = 8;
  int ty = 8;

  dim3 blocks((nx+tx-1)/tx, (ny+ty-1)/ty);
  dim3 threads(tx, ty);

  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx*ny*sizeof(curandState)));

  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::vector<vec3> point_list;
  // point_list.push_back(vec3(0.0f, 0.0f, 0.0f));
  // point_list.push_back(vec3(3.0f, 0.0f, 0.0f));
  // point_list.push_back(vec3(0.0f, 3.0f, 0.0f));
  // point_list.push_back(vec3(0.0f, 0.0f, 3.0f));
  std::vector<int> triangle_list;
  // triangle_list.push_back(0);
  // triangle_list.push_back(1);
  // triangle_list.push_back(2);
  // triangle_list.push_back(0);
  // triangle_list.push_back(2);
  // triangle_list.push_back(3);
  // triangle_list.push_back(0);
  // triangle_list.push_back(3);
  // triangle_list.push_back(1);
  // triangle_list.push_back(1);
  // triangle_list.push_back(3);
  // triangle_list.push_back(2);

  load_obj("../models/teapot.obj", point_list, triangle_list);

  int num_triangles = triangle_list.size()/3;
  int num_points = point_list.size();

  vec3 *d_point_list;
  checkCudaErrors(cudaMalloc((void**)&d_point_list, num_points*sizeof(vec3)));
  int *d_triangle_list;
  checkCudaErrors(cudaMalloc((void**)&d_triangle_list, num_triangles*3*sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_point_list, point_list.data(), num_points*sizeof(vec3), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_triangle_list, triangle_list.data(),
			     num_triangles*3*sizeof(int), cudaMemcpyHostToDevice));

  int num_spheres = 480;  

  Object **d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, (num_spheres + 1)*sizeof(Object*)));
  Object **d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Object*)));

  create_world<<<1, 1>>>(d_list, d_world, num_spheres + 1, d_rand_state,
			 d_point_list, num_points, d_triangle_list, num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(raw_image, nx, ny, num_samples, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_list, d_world, num_spheres + 1);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(d_point_list));
  checkCudaErrors(cudaFree(d_triangle_list));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_rand_state));
}


int main(int argc, char** argv)
{
  std::string output = "";
  int nx;
  int ny;
  int ns;
  
  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("output,o", po::value<std::string>(&output)->required(), "Filename for the output figure")
      ("resolution-x,x", po::value<int>(&nx)->default_value(1600), "Horizontal output resolution")
      ("resolution-y,y", po::value<int>(&ny)->default_value(800), "Vertical output resolution")
      ("num-samples,s", po::value<int>(&ns)->default_value(100), "Number of samples per pixel");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(vm);
  }
  catch(const std::runtime_error &ex) {
    std::cerr << ex.what() << std::endl;
  }

  int num_pixels = nx*ny;

  size_t fb_size = num_pixels*sizeof(vec3);
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  generate_test_image(fb, nx, ny, ns);
  write_ppm(fb, nx, ny, output);

  cudaFree(fb);

  return 0;
}
