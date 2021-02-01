#include <iostream>

#include "Logger.h"
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

  initLogger();
  BOOST_LOG_TRIVIAL(trace) << "This is a trace severity message";
  BOOST_LOG_TRIVIAL(debug) << "This is a debug severity message";
  BOOST_LOG_TRIVIAL(info) << "This is an informational severity message";
  BOOST_LOG_TRIVIAL(warning) << "This is a warning severity message";
  BOOST_LOG_TRIVIAL(error) << "This is an error severity message";
  BOOST_LOG_TRIVIAL(fatal) << "and this is a fatal severity message";

  SceneObjects sceneObjects;
  SceneParser sceneParser(programArgs.SceneFilePath, sceneObjects, status);
  exitIfError(status);

  // load scene objects to object list
  // pass those objects to device
  // render scene
  // save the rendered image to file

  return 0;
}
