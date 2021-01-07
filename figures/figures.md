@startuml
abstract Object {
 +hit() : bool
 +getBBox() : bool
 Material* material 
 AABB* bbox
}
class World {
 +hit() : bool
 +getBBox() : bool
}
class Mesh{
 +hit() : bool
 +getBBox() : bool 
}
class Sphere{
 +hit() : bool
 +getBBox() : bool 
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

Object *.. Material

class GenericObject

Mesh <.. GenericObject
Sphere <.. GenericObject

class GenericMaterial

Lambertian <.. GenericMaterial
Metal <.. GenericMaterial
Dielectric <.. GenericMaterial

class Renderer {
 +render(in World* world, Camera* camera)
}

Object <.. Renderer

class Camera

Camera <.. Renderer

class SceneDescriptorParser

class WorldCreator

class CommandLineArgumentParser

class SceneParameters

@enduml

