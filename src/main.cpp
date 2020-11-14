#include "arg_parser.h"

int main(int argc, char** argv)
{
  ProgramArgs args;
  // parse command line arguments
  parse_args(argc, argv, args);

  // load scene objects to object list
  // pass those objects to device
  // render scene
  // save the rendered image to file

  return 0;
}
