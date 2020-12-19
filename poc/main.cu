#include <cstdio>
#include <cstring>
#include <iostream>

enum class MaterialType { None, Lambertian, Metal };


struct vec3 {
  float x;
  float y;
  float z;
};


struct GenericMaterial {
  __device__ __host__ GenericMaterial():
    matType(MaterialType::None),
    numScalars(0), scalars(nullptr),
    numVectors(0), vectors(nullptr) {
    printf("Constructing GenericMaterial\n");
  }

  MaterialType matType;
  int numScalars;
  float *scalars;
  int numVectors;
  vec3 *vectors;
};


class Material {
public:
  __device__ virtual bool scatter() const = 0;
};


class Lambertian: public Material {
public:
  __device__ Lambertian(GenericMaterial* genMat) {
    m_albedo = genMat->vectors[0];
  }
  __device__ virtual bool scatter() const {
    printf("Lambertian albedo %f %f %f\n",
	   m_albedo.x, m_albedo.y, m_albedo.z);
    return true;
  }
  const MaterialType matType = MaterialType::Lambertian;
private:
  vec3 m_albedo;
};


class Metal: public Material {
public:
  __device__ Metal(GenericMaterial* genMat) {
    m_fuzz = genMat->scalars[0];
    m_albedo = genMat->vectors[0];
  }
  __device__ virtual bool scatter() const {
    printf("Metal albedo %f, %f %f %f\n",
	   m_fuzz, m_albedo.x, m_albedo.y, m_albedo.z);
    return true;
  }
  const MaterialType matType = MaterialType::Metal;
private:
  float m_fuzz;
  vec3 m_albedo;
};


enum class ObjectType { None, Mesh, Sphere };


struct GenericObject {
  __device__ __host__ GenericObject() :
    objType(ObjectType::None),
    numScalars(0), scalars(nullptr),
    numVectors(0), vectors(nullptr) {}

  ObjectType objType;
  int numScalars;
  float *scalars;
  int numVectors;
  vec3 *vectors;
};


class Object {
public:
  __device__ virtual bool hit() const = 0;

  const ObjectType objType = ObjectType::None;
protected:
  Material *m_material;
};


class Mesh: public Object {
public:
  const ObjectType objType = ObjectType::Mesh;
  __device__ Mesh(GenericObject* genObj, Material* mat) {
    m_numVertices = genObj->numVectors;
    m_vertices = genObj->vectors;
    m_material = mat;
  }
  __device__ virtual bool hit() const {
    printf("hitting trianglemesh %d\n", m_numVertices);
    m_material->scatter();
    return true;
  }
private:
  int m_numVertices;
  vec3 *m_vertices;
};


class Sphere: public Object {
public:
  const ObjectType objType = ObjectType::Sphere;
  __device__ Sphere(GenericObject* genObj, Material* mat) {
    m_radius = genObj->scalars[0];
    m_x = genObj->scalars[1];
    m_material = mat;
  }
  __device__ virtual bool hit() const {
    printf("hitting sphere %f, %f\n", m_radius, m_x);
    m_material->scatter();    
    return true;
  }
private:
  float m_radius;
  float m_x;
};


class ObjectList: public Object {
 public:
  __device__ ObjectList() :
    objects(nullptr), num_objects(0) {}
  __device__ ObjectList(Object** objects, int num_objects) :
    objects(objects), num_objects(num_objects) {}

  __device__ virtual bool hit() const {
    for (int i = 0; i < num_objects; i++) {
      objects[i]->hit();
    }
    return true;
  }

  Object **objects;
  int num_objects;
};


__global__ void create_world(GenericObject* genObjList,
			     GenericMaterial* genMatList,
			     int numObjects,
			     Object** objectList,
			     Object** world)
{
  printf("numObjects = %d\n", numObjects);
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {

    Material *mat = nullptr;
    
    GenericMaterial* genMat = &(genMatList[objIdx]);
    if (genMat->matType == MaterialType::Lambertian) {
      printf("Lambertian\n");
      mat = new Lambertian(genMat);
    }
    else if (genMat->matType == MaterialType::Metal) {
      printf("Metal\n");
      mat = new Metal(genMat);
    }

    GenericObject* genObj = &(genObjList[objIdx]);
    if (genObj->objType == ObjectType::Sphere) {
      printf("Sphere %f %f\n", genObj->scalars[0], genObj->scalars[1]);
      objectList[objIdx] = new Sphere(genObj, mat);
    }
    else if (genObj->objType == ObjectType::Mesh) {
      printf("Mesh %d\n", genObj->numVectors);
      objectList[objIdx] = new Mesh(genObj, mat);
    }

  }
  *world = new ObjectList(objectList, numObjects);
}


__global__ void render_world(Object** world, int numObjects)
{
  (*world)->hit();
}

void setupScene(GenericObject** genObjList,
		GenericMaterial** genMatList, int& o_numObjects)
{
  std::cout << "Setting up scene ..." << std::endl;
  int numObjects = 2;
  *genObjList = new GenericObject[numObjects];
  *genMatList = new GenericMaterial[numObjects];  

  // Sphere
  int numScalars = 2;
  (*genObjList)[0].objType = ObjectType::Sphere;
  (*genObjList)[0].numScalars = numScalars;
  (*genObjList)[0].scalars = new float[numScalars];
  (*genObjList)[0].scalars[0] = 1.0f;
  (*genObjList)[0].scalars[1] = 0.5f;
  (*genMatList)[0].matType = MaterialType::Lambertian;
  (*genMatList)[0].numVectors = 1;
  (*genMatList)[0].vectors = new vec3;
  (*genMatList)[0].vectors[0] = {0.25f, 0.35f, 0.45f};

  // Mesh
  int numVectors = 3;
  (*genObjList)[1].objType = ObjectType::Mesh;
  (*genObjList)[1].numVectors = numVectors;
  (*genObjList)[1].vectors = new vec3[numVectors];
  (*genObjList)[1].vectors[0] = {0.0f, 0.0f, 0.0f};
  (*genObjList)[1].vectors[1] = {1.0f, 0.0f, 0.0f};
  (*genObjList)[1].vectors[2] = {0.0f, 1.0f, 0.0f};
  (*genMatList)[1].matType = MaterialType::Metal;
  (*genMatList)[1].numVectors = 1;
  (*genMatList)[1].vectors = new vec3;
  (*genMatList)[1].vectors[0] = {0.55f, 0.65f, 0.75f};
  (*genMatList)[1].numScalars = 1;  
  (*genMatList)[1].scalars = new float;
  (*genMatList)[1].scalars[0] = 0.95f;

  o_numObjects = numObjects;
  std::cout << "Setting up scene done." << std::endl;  
}

int main()
{
  GenericObject *genObjList;
  GenericMaterial *genMatList;
  int numObjects;
  setupScene(&genObjList, &genMatList, numObjects);
  
  // copy genObjList to device
  // helpful: https://stackoverflow.com/questions/19404965/how-to-use-cudamalloc-cudamemcpy-for-a-pointer-to-a-structure-containing-point
  GenericObject *h_genObjList = new GenericObject[numObjects];
  std::memcpy(h_genObjList, genObjList, numObjects*sizeof(GenericObject));
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {

    if (h_genObjList[objIdx].numScalars > 0) {
      cudaMalloc((void**)&(h_genObjList[objIdx].scalars),
  		 h_genObjList[objIdx].numScalars*sizeof(float));
      cudaMemcpy(h_genObjList[objIdx].scalars, genObjList[objIdx].scalars,
		 h_genObjList[objIdx].numScalars*sizeof(float), cudaMemcpyHostToDevice);
    }
    if (h_genObjList[objIdx].numVectors > 0) {
      cudaMalloc((void**)&(h_genObjList[objIdx].vectors),
  		 h_genObjList[objIdx].numVectors*sizeof(vec3));
      cudaMemcpy(h_genObjList[objIdx].vectors, genObjList[objIdx].vectors,
		 h_genObjList[objIdx].numVectors*sizeof(vec3), cudaMemcpyHostToDevice);
    }
  }
  GenericObject *d_genObjList;
  cudaMalloc((void**)&d_genObjList, numObjects*sizeof(GenericObject));
  cudaMemcpy(d_genObjList, h_genObjList, numObjects*sizeof(GenericObject), cudaMemcpyHostToDevice);
  //~copy genObjList to device

  // copy genMatList to device
  // helpful: https://stackoverflow.com/questions/19404965/how-to-use-cudamalloc-cudamemcpy-for-a-pointer-to-a-structure-containing-point
  GenericMaterial *h_genMatList = new GenericMaterial[numObjects];
  std::memcpy(h_genMatList, genMatList, numObjects*sizeof(GenericMaterial));
  for (int objIdx = 0; objIdx < numObjects; objIdx++) {

    if (h_genMatList[objIdx].numScalars > 0) {
      cudaMalloc((void**)&(h_genMatList[objIdx].scalars),
  		 h_genMatList[objIdx].numScalars*sizeof(float));
      cudaMemcpy(h_genMatList[objIdx].scalars, genMatList[objIdx].scalars,
		 h_genMatList[objIdx].numScalars*sizeof(float), cudaMemcpyHostToDevice);
    }
    if (h_genMatList[objIdx].numVectors > 0) {
      cudaMalloc((void**)&(h_genMatList[objIdx].vectors),
  		 h_genMatList[objIdx].numVectors*sizeof(vec3));
      cudaMemcpy(h_genMatList[objIdx].vectors, genMatList[objIdx].vectors,
		 h_genMatList[objIdx].numVectors*sizeof(vec3), cudaMemcpyHostToDevice);
    }
  }
  GenericMaterial *d_genMatList;
  cudaMalloc((void**)&d_genMatList, numObjects*sizeof(GenericMaterial));
  cudaMemcpy(d_genMatList, h_genMatList, numObjects*sizeof(GenericMaterial), cudaMemcpyHostToDevice);
  //~copy genMatList to device

  
  Object **d_list;
  cudaMalloc((void**)&d_list, numObjects*sizeof(Object*));
  Object **d_world;
  cudaMalloc((void**)&d_world, sizeof(Object*));

  create_world<<<1,1>>>(d_genObjList, d_genMatList, numObjects, d_list, d_world);
  cudaDeviceSynchronize();
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  render_world<<<1,1>>>(d_world, numObjects);
  cudaDeviceSynchronize();
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  return 0;
}
