#ifndef SCENE_DEVICE_CUH
#define SCENE_DEVICE_CUH

class MaterialFacroty {
  Material* createMaterial(GenericMaterialDevice* genMatDev) {

    switch (genMatDev->materialType) {
    case MaterialType::Lambertian:
      return new Lambertian();
      break;
    case MaterialType::Metal:
      return new Metal();
      break;
    default:
      break;
    }
  }
};

class ObjectFacroty {
  Object* createObject(GenericObjectDevice* genObjDev) {

    switch (genObjDev->objectType) {
    case ObjectType::Mesh:
      return new Mesh();
      break;
    case ObjectType::Sphere:
      return new Sphere();
      break;
    default:
      break;
    }
  }
};

class SceneDevice {

  
  
};

#endif//SCENE_DEVICE_CUH