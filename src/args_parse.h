#ifndef ARGS_PARSE_H
#define ARGS_PARSE_H

#include <string>
#include <boost/log/trivial.hpp>

#include "StatusCode.h"

struct ProgramArgs {
  std::string sceneFilePath;
  std::string pictureFilePath;
  int sampleCount;
  int imageWidth;
  int imageHeight;
  int rayDepth;
  boost::log::trivial::severity_level logLevel;
};

ProgramArgs parseArgs(int argc, char** argv, StatusCode& status);

#endif//ARGS_PARSE_H
