#include <iostream>

#include "logging.h"
#include "ArgumentParser.h"
#include "StatusCodes.h"
#include "scene_parse.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"
#include "Renderer.cuh"

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
  parseScene(programArgs.SceneFilePath, sceneRawObjects, status);
  exitIfError(status);

  // construct scene on device using class hierarchy
  // for objects and materials
  SceneDevice sceneDevice;
  sceneDevice.constructScene(sceneRawObjects, status);
  exitIfError(status);

  // render scene
  Renderer renderer;
  renderer.renderScene(sceneDevice, status);
  exitIfError(status);

  // // save the rendered image to file
  // PictureSaver pictureSaver(render.getImageOnHost(status));
  // exitIfError(status);

  return 0;
}
