#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>

struct ProgramArgs {
  std::string scene;
  std::string output;
  int num_samples;
};

// Parse arguments from command line
void parse_args(int argc, char** argv, ProgramArgs& args);  


#endif//ARG_PARSER_H
