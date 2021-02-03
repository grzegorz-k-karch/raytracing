#include <iostream>

#include "logging.h"
#include "ArgumentParser.h"
#include "SceneParser.h"
#include "StatusCodes.h"

int main(int argc, char** argv)
{
  StatusCodes status = StatusCodes::NoError;

  // parse command line arguments
  ProgramArgs programArgs;
  parseArgs(argc, argv, programArgs, status);
  exitIfError(status);

  initLogger(programArgs.logLevel);

  SceneObjects sceneObjects;
  SceneParser sceneParser(programArgs.SceneFilePath, sceneObjects, status);
  exitIfError(status);

  // load scene objects to object list
  // pass those objects to device
  // render scene
  // save the rendered image to file

  return 0;
}
