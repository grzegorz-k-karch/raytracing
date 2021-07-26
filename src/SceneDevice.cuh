#ifndef SCENE_DEVICE_CUH
#define SCENE_DEVICE_CUH

#include <assert.h>

#include "SceneRawObjects.h"
#include "Objects.cuh"
#include "Materials.cuh"
#include "Textures.cuh"

class ObjectFactory {
public:
  __device__  
  static Object* createObject(const GenericObjectDevice* genObjDev) {

    Object *obj = nullptr;
    switch (genObjDev->m_objectType) {
    case ObjectType::Mesh:
      obj = new Mesh(genObjDev);
      break;
    case ObjectType::Sphere:
      obj = new Sphere(genObjDev);
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
  SceneDevice() :
    m_camera(nullptr),
    m_world(nullptr) {}
  ~SceneDevice();
  void constructScene(SceneRawObjects& sceneRawObjects,
		      StatusCodes& status);  
  Camera *m_camera;
  Object** m_world;
};

#endif//SCENE_DEVICE_CUH
