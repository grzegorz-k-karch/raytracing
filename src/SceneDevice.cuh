#ifndef SCENE_DEVICE_CUH
#define SCENE_DEVICE_CUH

#include "SceneRawObjects.h"
#include "Objects.cuh"
#include "Materials.cuh"

// class MaterialFacroty {
// public:
//   static Material* createMaterial(GenericMaterialDevice* genMatDev) {

//     switch (genMatDev->materialType) {
//     case MaterialType::Lambertian:
//       return new Lambertian();
//       break;
//     case MaterialType::Metal:
//       return new Metal();
//       break;
//     default:
//       break;
//     }
//   }
// };


// class ObjectFacroty {
// public:
//   static Object* createObject(GenericObjectDevice* genObjDev) {

//     switch (genObjDev->objectType) {
//     case ObjectType::Mesh:
//       return new Mesh();
//       break;
//     case ObjectType::Sphere:
//       return new Sphere();
//       break;
//     default:
//       break;
//     }
//   }
// };

class SceneDevice {
public:
  SceneDevice(const SceneRawObjectsDevice* sceneRawObjectsDevice);
};

#endif//SCENE_DEVICE_CUH
