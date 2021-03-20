#include <iostream>

#include "logging.h"
#include "ArgumentParser.h"
#include "StatusCodes.h"
#include "scene_parse.h"
#include "SceneObjects.h"

int main(int argc, char** argv)
{
  StatusCodes status = StatusCodes::NoError;

  // parse command line arguments
  ProgramArgs programArgs;
  parseArgs(argc, argv, programArgs, status);
  exitIfError(status);

  initLogger(programArgs.logLevel);

  // get all objects into SceneObjects struct
  SceneObjects sceneObjects;
  parseScene(programArgs.SceneFilePath, sceneObjects, status);
  exitIfError(status);

  // pass those objects to device
  SceneObjectsDevice *sceneObjectsDevice;
  sceneObjects.copyToDevice(&sceneObjectsDevice, status);
  exitIfError(status);

  // construct scene on device


  // render scene


  // save the rendered image to file

  return 0;
}
