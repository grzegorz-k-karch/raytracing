#ifndef SCENE_DEVICE_CUH
#define SCENE_DEVICE_CUH

#include <assert.h>

#include "SceneRawObjects.h"
#include "Objects.cuh"
#include "Materials.cuh"
#include "Textures.cuh"

class MaterialFactory {
public:
  __device__
  static Material* createMaterial(const GenericMaterialDevice* genMatDev) {

    Material *mat = nullptr;
    switch (genMatDev->materialType) {
    case MaterialType::DiffuseLight:
      mat = new DiffuseLight(genMatDev);
      break;
    case MaterialType::Lambertian:
      mat = new Lambertian(genMatDev);
      break;
    case MaterialType::Metal:
      mat = new Metal(genMatDev);
      break;
    case MaterialType::Dielectric:
      mat = new Dielectric(genMatDev);
      break;
    case MaterialType::Parametric:
      mat = new Parametric(genMatDev);
      break;
    case MaterialType::None:
      break;
    default:
      break;
    }
    assert(mat != nullptr);
    return mat;
  }
};

class ObjectFactory {
public:
  __device__  
  static Object* createObject(const GenericObjectDevice* genObjDev,
			      const Material* material) {

    Object *obj = nullptr;
    switch (genObjDev->objectType) {
    case ObjectType::Mesh:
      obj = new Mesh(genObjDev, material);
      break;
    case ObjectType::Sphere:
      obj = new Sphere(genObjDev, material);
      break;
    case ObjectType::None:
      break;
    default:
      break;
    }
    assert(obj != nullptr);    
    return obj;
  }
};

class SceneDevice {
public:
  void constructScene(const SceneRawObjects& sceneRawObjects,
		      StatusCodes& status);  
  Camera *m_camera;
  Object** m_world;
};

#endif//SCENE_DEVICE_CUH
