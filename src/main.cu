#include "gkk_cuda_utils.cuh"
#include "gkk_vec.cuh"
#include "gkk_color.cuh"
#include "gkk_object.cuh"
#include "gkk_geometry.cuh"
#include "gkk_camera.cuh"
#include "gkk_xmlreader.h"
#include "gkk_aabb.cuh"
#include "gkk_material.cuh"

#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

namespace po = boost::program_options;


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
      color = vec3(std::sqrt(color.r()),
		   std::sqrt(color.g()),
		   std::sqrt(color.b()));
      int ir = int(255.99f*color.r());
      int ig = int(255.99f*color.g());
      int ib = int(255.99f*color.b());
      fs << ir << " " << ig << " " << ib << "\n";
    }
  }
  fs.close();
  return 0;
}

__global__ void init_rand_state(int max_x, int max_y, curandState* rand_state)
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
		       Object** world, curandState* rand_state,
		       Camera* camera)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pixel_idx = i + j*max_x;
  curandState local_rand_state = rand_state[pixel_idx];
  vec3 color = vec3(0.0f, 0.0f, 0.0f);

  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state))/float(max_x);
    float v = float(j + curand_uniform(&local_rand_state))/float(max_y);
    Ray ray = camera->get_ray(u, v, &local_rand_state);
    color += get_color(ray, *world, &local_rand_state);
  }
  fb[pixel_idx] = color/float(ns);
}

__global__
void create_world(Object** obj_list, Object** world, int n,
		  curandState* rand_state){
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = rand_state[0];

    obj_list[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
    			     new Lambertian(vec3(0.5f, 0.5f, 0.5f)));
    obj_list[1] = new Sphere(vec3(-8.0f, 1.0f, -1.0f), 1.0f,
    			     new Dielectric(1.5f));
    obj_list[2] = new Sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f,
    			     new Lambertian(vec3(0.4f, 0.2f, 0.1f)));
    obj_list[3] = new Sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f,
    			     new Metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    for (int i = 0; i < 5; i++) {
      printf("|||| %p %p %f\n", obj_list[i], ((Sphere*)obj_list[i])->material);
      ((Sphere*)obj_list[i])->material->d_print();
    }

    *world = new ObjectList(obj_list, n);
  }
}


__global__ void destroy_world(Object** d_list, Object** d_world, int n)
{
  for (int i = 0; i < n; i++) {
    delete d_list[i];
  }
  delete *d_world;
}


void generate_test_image(vec3* raw_image,
			 const int nx, const int ny,
			 const int num_samples,
			 const Camera& camera,
			 const TriangleMesh& triangle_mesh,
			 const Sphere& sphere)
{
  int tx = 8;
  int ty = 8;

  dim3 blocks((nx+tx-1)/tx, (ny+ty-1)/ty);
  dim3 threads(tx, ty);

  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx*ny*sizeof(curandState)));
  init_rand_state<<<blocks, threads>>>(nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  Camera *d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
  checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera),
			     cudaMemcpyHostToDevice));  

  int num_spheres = 4;
  int num_objects = num_spheres + 1;

  Object **d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, num_objects*sizeof(Object*)));
  Object **d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Object*)));

  // triangle_mesh.copy_to_device(d_list, num_spheres);
  sphere.copy_to_device(d_list, num_spheres);

  create_world<<<1, 1>>>(d_list, d_world, num_objects, d_rand_state); 

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(raw_image, nx, ny, num_samples,
			      d_world, d_rand_state, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  destroy_world<<<1, 1>>>(d_list, d_world, num_spheres);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_camera));  
}

int main(int argc, char** argv)
{
  std::string output = "";
  std::string config_filepath = "";  
  int ns;

  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("output,o", po::value<std::string>(&output)->required(),
       "Filename for the output figure")
      ("num-samples,s", po::value<int>(&ns)->default_value(64),
       "Number of samples per pixel")
      ("config", po::value<std::string>(&config_filepath)->required(),
       "File with scene config");
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

  pt::ptree scene_tree;
  xml_read(config_filepath, scene_tree);
  
  pt::ptree camera_tree = scene_tree.get_child("scene.camera");
  Camera camera(camera_tree);

  pt::ptree triangle_mesh_tree = scene_tree.get_child("scene.triangle_mesh");
  TriangleMesh triangle_mesh(triangle_mesh_tree);

  pt::ptree sphere_tree = scene_tree.get_child("scene.sphere");
  Sphere sphere(sphere_tree);

  int nx = camera.res_x;
  int ny = camera.res_y;
  int num_pixels = nx*ny;
  size_t framebuffer_size = num_pixels*sizeof(vec3);

  vec3 *framebuffer;
  checkCudaErrors(cudaMallocManaged((void**)&framebuffer, framebuffer_size));

  generate_test_image(framebuffer, nx, ny, ns, camera, triangle_mesh, sphere);

  write_ppm(framebuffer, nx, ny, output);

  cudaFree(framebuffer);

  return 0;
}
