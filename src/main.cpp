#include <iostream>
#include <vector>
#include <limits>

#include "gkk_vec.h"
#include "gkk_ray.h"
#include "gkk_material.h"
#include "gkk_geometry.h"
#include "gkk_object.h"
#include "gkk_camera.h"
#include "gkk_random.h"

vec3 get_plane_color(const Ray& ray)
{
  vec3 unit_direction = normalize(ray.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f - t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
}

vec3 get_color(const Ray& ray, Object* world, int depth)
{
  hit_record hrec;
  vec3 color;
  if (world->hit(ray, 0.001f, std::numeric_limits<float>::max(), hrec)) {
    Ray scattered;
    vec3 attenuation;
    if (depth < 50 && hrec.material_ptr->scatter(ray, hrec, attenuation, scattered)) {
      color = attenuation*get_color(scattered, world, depth+1);
    }
  }
  else {
    color = get_plane_color(ray);
  }
  return color;
}

// assuming pixel values are in range (0,1)
int write_ppm(const std::vector<vec3>& raw_image,
	      const int nx=300,
	      const int ny=200)
{
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      int pixel_idx = i + j*nx;
      vec3 color = raw_image[pixel_idx];
      // gamma correction
      color = vec3(std::sqrt(color.r()), std::sqrt(color.g()), std::sqrt(color.b()));
      int ir = int(255.99*color.r());
      int ig = int(255.99*color.g());
      int ib = int(255.99*color.b());
      std::cout << ir << " " << ig << " " << ib << std::endl;
    }
  }
  return 0;
}

Object* random_scene()
{
  int n = 500;
  Object **objects = new Object*[n+1];
  objects[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
		       new Lambertian(vec3(0.5f, 0.5f, 0.5f)));
  int i = 1;
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = gkk_random<float>();
      vec3 center(a + 0.9f*gkk_random<float>(),
		  0.2f,
		  b + 0.9f*gkk_random<float>());
      if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
	if (choose_mat < 0.8f) { // diffuse
	  objects[i++] = new Sphere(center, 0.2f,
				 new Lambertian(vec3(gkk_random<float>()*gkk_random<float>(),
						     gkk_random<float>()*gkk_random<float>(),
						     gkk_random<float>()*gkk_random<float>())));
	}
	else if (choose_mat < 0.95f) { // metal
	  objects[i++] = new Sphere(center, 0.2f,
				 new Metal(vec3(0.5f*(1.0f + gkk_random<float>()),
						0.5f*(1.0f + gkk_random<float>()),
						0.5f*(1.0f + gkk_random<float>())),
					   0.5f*gkk_random<float>()));
	}
	else { // glass
	  objects[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
	}
      }
    }
  }

  objects[i++] = new Sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
  objects[i++] = new Sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(vec3(0.4f, 0.2f, 0.1f)));
  objects[i++] = new Sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

  return new ObjectList(objects, i);
}

int generate_test_image(std::vector<vec3>& raw_image,
			const int nx=400,
			const int ny=200)
{
  // default camera
  // Camera camera(90.0f, float(nx)/float(ny));
  vec3 lookfrom = vec3(-1.0f, 2.0f, 4.0f);
  vec3 lookat = vec3(0.0f, 1.0f, 0.0f);
  vec3 up = vec3(0.0f, 1.0f, 0.0f);
  Camera camera(lookfrom, lookat, up, 60.0f, float(nx)/float(ny),
		0.5f, (lookfrom-lookat).length());  
  
  Object *world = random_scene();

  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      
      vec3 color = vec3(0.0f, 0.0f, 0.0f);
      int ns = 500;
      for (int s = 0; s < ns; s++) {
	float u = float(i + gkk_random<float>())/float(nx);
	float v = float(j + gkk_random<float>())/float(ny);
	Ray ray = camera.get_ray(u, v);

	color += get_color(ray, world, 0);
      }

      int pixel_idx = i + j*nx;
      raw_image[pixel_idx] = color/ns;
      if (pixel_idx % 1000 == 0) {
	std::cerr << "pixel_idx " << pixel_idx << std::endl;
      }
    }
  }
}

int main()
{
  int nx = 1600;
  int ny = 800;
  std::vector<vec3> raw_image(nx*ny);

  generate_test_image(raw_image, nx, ny);
  write_ppm(raw_image, nx, ny);

  return 0;
}
