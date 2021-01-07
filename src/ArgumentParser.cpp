#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "ArgumentParser.h"
#include "StatusCodes.h"

namespace po = boost::program_options;

std::ostream& exception_header(std::ostream& os)
{
  os << "[Program option exception]: ";
  return os;
}

void parseArgsFromCmdLine(int argc, char** argv, ProgramArgs& args,
			  StatusCodes& status)
{
  status = StatusCodes::NoError;
  std::string configFilePath;
  try {
    // add options for cmd line only
    po::options_description generic("Generic options");
    generic.add_options()
      ("help,h", "Help screen")
      ("config,c",
       po::value<std::string>(&configFilePath),
       "Filename for the program config file")
      ;

    // add options for both cmd line and config file
    po::options_description config("Configuration");
    config.add_options()
      ("scene",
       po::value<std::string>(&args.SceneFilePath)->default_value(""),
       "File with scene description")
      ("num-samples,s",
       po::value<int>(&args.SampleCount)->default_value(64),
       "Number of samples per pixel")
      ("output,o",
       po::value<std::string>(&args.PictureFilePath)->default_value(""),
       "Filename for the output picture")
      ;

    po::options_description cmdLineOptions;
    cmdLineOptions.add(generic).add(config);
    po::options_description configFileOptions;
    configFileOptions.add(config);


    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(cmdLineOptions).run(), vm);

    if (vm.count("help")) {
      std::cout << cmdLineOptions << std::endl;
      status = StatusCodes::NoError;
      exit(EXIT_SUCCESS);
    }

    po::notify(vm);

    std::ifstream ifs(configFilePath.c_str());
    if (!ifs) {
      std::cerr << "Cannot open config file: " << configFilePath << std::endl;
      status = StatusCodes::FileError;
      return;
    }
    else {
      po::store(po::parse_config_file(ifs, configFileOptions), vm);
      po::notify(vm);
    }
  }
  catch(const po::required_option& ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
    status = StatusCodes::CmdLineError;
  }
  catch(const po::unknown_option& ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
    status = StatusCodes::CmdLineError;
  }
  catch(const po::error& ex) {
    exception_header(std::cerr);
    std::cerr << ex.what() << std::endl;
    status = StatusCodes::CmdLineError;
  }
}

void parseArgsFromFile(const std::string configFile)
{

}

void parseArgs(int argc, char** argv, ProgramArgs& args,
	       StatusCodes& status)
{
  parseArgsFromCmdLine(argc, argv, args, status);
}

void ProgramArgs::Print()
{
  std::cout << "ProgramArgs instance: " << std::endl
	    << "\tscene description: " << SceneFilePath << std::endl
	    << "\toutput picture:    " << PictureFilePath << std::endl
	    << "\tnumber of samples: " << SampleCount << std::endl;
}
