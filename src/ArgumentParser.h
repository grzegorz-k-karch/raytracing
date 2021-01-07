#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <string>
#include "StatusCodes.h"

struct ProgramArgs {
  std::string SceneFilePath;
  std::string PictureFilePath;
  int SampleCount;
  void Print();
};

// Parse arguments from command line
// argc: in
// argv: in
// args: out
void parseArgs(int argc, char** argv, ProgramArgs& args,
	       StatusCodes& status);


#endif//ARGUMENT_PARSER_H