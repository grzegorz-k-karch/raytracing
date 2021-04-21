#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <string>
#include <boost/log/trivial.hpp>

#include "StatusCodes.h"

struct ProgramArgs {
  std::string sceneFilePath;
  std::string pictureFilePath;
  int sampleCount;
  int imageWidth;
  int imageHeight;
  boost::log::trivial::severity_level logLevel;
  void Print();
};

// Parse arguments from command line
// argc: in
// argv: in
// args: out
void parseArgs(int argc, char** argv, ProgramArgs& args,
	       StatusCodes& status);


#endif//ARGUMENT_PARSER_H
