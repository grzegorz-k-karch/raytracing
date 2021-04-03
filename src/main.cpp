#include <iostream>

#include "logging.h"
#include "ArgumentParser.h"
#include "StatusCodes.h"
#include "scene_parse.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"

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

  // pass those objects to device
  SceneRawObjectsDevice *sceneRawObjectsDevice;
  sceneRawObjects.copyToDevice(&sceneRawObjectsDevice, status);
  exitIfError(status);

  // construct scene on device using class hierarchy
  // for objects and materials
  SceneDevice sceneDevice(sceneRawObjectsDevice);


  // render scene
  

  // save the rendered image to file

  return 0;
}
