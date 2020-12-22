@startuml
abstract Object {
 +hit() : bool
 +getBBox() : bool
}
class World {
 +hit() : bool
 +getBBox() : bool
 AABB* bbox
}
class Mesh{
 +hit() : bool
 +getBBox() : bool 
 Material* material
 AABB* bbox 
}
class Sphere{
 +hit() : bool
 +getBBox() : bool 
 Material* material
 AABB* bbox 
}

World o-- Object
Object <|-- World
Object <|-- Mesh
Object <|-- Sphere

abstract Material {
 +scatter()
}
class Lambertian {
 +scatter()
}
class Metal {
 +scatter()
}
class Dielectric {
 +scatter()
}

Material <|-- Lambertian
Material <|-- Metal
Material <|-- Dielectric

class GenericObject

Mesh <.. GenericObject
Sphere <.. GenericObject


class GenericMaterial

Lambertian <.. GenericMaterial
Metal <.. GenericMaterial
Dielectric <.. GenericMaterial

class Renderer {
 +render(World* world, Camera* camera)
}

Object <.. Renderer

class Camera

Camera <.. Renderer

@enduml

