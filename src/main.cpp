#include <iostream>

#include "ArgumentParser.h"
#include "SceneParser.h"
#include "StatusCodes.h"

int main(int argc, char** argv)
{
  StatusCodes status = StatusCodes::NoError;

  ProgramArgs programArgs;

  // parse command line arguments
  parseArgs(argc, argv, programArgs, status);
  exitIfError(status);

  SceneParser sceneParser(programArgs.SceneFilePath);

  // load scene objects to object list
  // pass those objects to device
  // render scene
  // save the rendered image to file

  return 0;
}
