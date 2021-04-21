#include <iostream>

#include "logging.h"
#include "ArgumentParser.h"
#include "StatusCodes.h"
#include "scene_parse.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"
#include "Renderer.cuh"
#include "ImageSaver.h"

#include <vector> // TODO delete

int main(int argc, char** argv)
{
  StatusCodes status = StatusCodes::NoError;

  // parse command line arguments
  ProgramArgs programArgs;
  parseArgs(argc, argv, programArgs, status);
  exitIfError(status);

  initLogger(programArgs.logLevel);

  // get all objects into SceneRawObjects struct
  SceneRawObjects sceneRawObjects;
  parseScene(programArgs.sceneFilePath, sceneRawObjects, status);
  exitIfError(status);

  // construct scene on device using class hierarchy
  // for objects and materials
  SceneDevice sceneDevice;
  sceneDevice.constructScene(sceneRawObjects, status);
  exitIfError(status);

  Renderer renderer(programArgs.imageWidth, programArgs.imageHeight,
		    programArgs.sampleCount);
  // initialize random state and image buffer
  renderer.initBuffers(status);
  exitIfError(status);
  // render scene
  renderer.renderScene(sceneDevice, status);
  exitIfError(status);

  std::vector<float3> image;
  renderer.getImageOnHost(image, status);

  ImageSaver imageSaver;
  // save the rendered image to file
  imageSaver.saveImage(image, programArgs.imageWidth,
		       programArgs.imageHeight,
		       programArgs.pictureFilePath,
		       status);
  exitIfError(status);

  return 0;
}
