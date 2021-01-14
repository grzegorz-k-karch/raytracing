#ifndef GENERIC_OBJECT_H
#define GENERIC_OBJECT_H

class GenericObject {
 public:
  GenericObject() :
    objType(ObjectType::None),
    numScalars(0), scalars(nullptr),
    numVectors(0), vectors(nullptr) {}
  
  ObjectType objType;
  int numScalars;
  float *scalars;
  int numVectors;
  vec3 *vectors;
};

class GenericMaterial {
 public:
  GenericMaterial():
    matType(MaterialType::None),
    numScalars(0), scalars(nullptr),
    numVectors(0), vectors(nullptr) {}

  MaterialType matType;
  int numScalars;
  float *scalars;
  int numVectors;
  vec3 *vectors;
};

#endif//GENERIC_OBJECT_H
