#include "arg_parser.h"

#include <iostream>
#include <string>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

std::ostream& exception_header(std::ostream& os)
{
  os << "[Program option exception]: ";
  return os;
}

void parse_args(int argc, char** argv, ProgramArgs& args)
{
  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("output,o",
       po::value<std::string>(&args.output)->required(),
       "Filename for the output picture")
      ("num-samples,s",
       po::value<int>(&args.num_samples)->default_value(64),
       "Number of samples per pixel")
      ("scene",
       po::value<std::string>(&args.scene)->required(),
       "File with scene description")
      ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return;
    }
    po::notify(vm);
  }
  catch(const po::required_option &ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
  }
  catch(const po::unknown_option &ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
  }
  catch(const std::runtime_error &ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
  }
}
