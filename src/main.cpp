#include <iostream>
#include "ArgumentParser.h"
#include "SceneParser.h"
#include "StatusCodes.h"

int main(int argc, char** argv)
{
  ProgramArgs programArgs;
  StatusCodes status = StatusCodes::NoError;

  // parse command line arguments
  parseArgs(argc, argv, programArgs, status);
  if (status != StatusCodes::NoError) {
    std::cerr << "Error: "
	      << static_cast<std::underlying_type<StatusCodes>::type>(status)
	      << std::endl
	      << "Exiting." << std::endl;
    return -1;
  }
  programArgs.Print();

  SceneParser sceneParser(programArgs.SceneFilePath);

  // load scene objects to object list
  // pass those objects to device
  // render scene
  // save the rendered image to file

  return 0;
}
