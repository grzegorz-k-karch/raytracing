#ifndef SCENE_RAW_OBJECTS_H
#define SCENE_RAW_OBJECTS_H

#include <vector>

#include "GenericObject.h"
#include "GenericMaterial.h"
#include "Camera.cuh"
#include "StatusCode.h"
#include "OptixRenderer.h"

#include "logging.h"

struct SceneRawObjectsDevice {

  SceneRawObjectsDevice() :
    m_camera(nullptr),
    m_objects(nullptr), m_numObjects(0) {}

  ~SceneRawObjectsDevice() {
  }

  Camera* m_camera;
  GenericObjectDevice* m_objects;
  int m_numObjects;
};


class SceneRawObjects {
public:
  // default constructor
  SceneRawObjects():
    m_inputFlags{OPTIX_GEOMETRY_FLAG_NONE} {}
  ~SceneRawObjects();

  StatusCode parseScene(const std::string filepath,
			std::vector<GenericObject>& objects);
  StatusCode loadScene(const std::string filepath);
  void setCamera(Camera&& camera) {
    m_camera = std::move(camera);
  }
  Camera getCamera() {
    return m_camera;
  }
  StatusCode copyToDevice(std::vector<GenericObject>& objects);
  void generateOptixBuildInput(GenericObjectDevice& genObjDev,
			       OptixBuildInput& buildInput);
  void generateTraversableHandles(OptixDeviceContext context,
				  std::vector<OptixTraversableHandle>& traversableHandles);
  void generateHitGroupRecords(std::vector<HitGroupSBTRecord>& hitgroupRecords);
  void buildAccelStruct(OptixDeviceContext context,
			OptixBuildInput* buildInput,
			OptixTraversableHandle* traversableHandle);

private:
  Camera m_camera;
  std::vector<GenericObjectDevice> m_objectsDevice;
  SceneRawObjectsDevice m_h_sceneRawObjectsDevice;
  std::vector<CUdeviceptr> m_d_outputBuffers;
  const uint32_t m_inputFlags[1];
};

#endif//SCENE_RAW_OBJECTS_H
