#include <unordered_set>

#include "logging.h"
#include "SceneRawObjects.h"

void SceneRawObjects::generateTraversableHandles(std::vector<OptixTraversableHandle>& traversableHandles)
{
  std::unordered_set<ObjectType> typesInScene;
  for (auto& object : m_objects) {
    OptixBuildInput buildInput;
    object.generateOptixBuildInput(buildInput);
    typesInScene.insert(object.getObjectType());
  }
  LOG_TRIVIAL(info) << "num types in scene: " << typesInScene.size() << "\n";
}
