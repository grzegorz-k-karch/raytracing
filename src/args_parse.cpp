#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "args_parse.h"
#include "StatusCode.h"

namespace po = boost::program_options;

void add_generic_options(po::options_description& generic,
			 std::string* configFilePath,
			 std::string* logLevel)
{
  generic.add_options()
    ("help,h", "Help screen")
    ("config,c",
     po::value<std::string>(configFilePath),
     "Filename for the program config file")
    ("log-level",
     po::value<std::string>(logLevel)->default_value("info"),
     "Logging level")
    ;
}

void add_config_options(po::options_description& config,
			ProgramArgs* args)
{
  config.add_options()
    ("scene",
     po::value<std::string>(&(args->sceneFilePath))->default_value(""),
     "File with scene description")
    ("num-samples,s",
     po::value<int>(&(args->sampleCount))->default_value(4),
     "Number of samples per pixel")
    ("ray-depth",
     po::value<int>(&(args->rayDepth))->default_value(32),
     "Number of samples per pixel")
    ("res-x,x",
     po::value<int>(&(args->imageWidth))->default_value(600),
     "Horizontal resolution")
    ("res-y,y",
     po::value<int>(&(args->imageHeight))->default_value(400),
     "Vertical resolution")
    ("output,o",
     po::value<std::string>(&(args->pictureFilePath))->default_value(""),
     "Filename for the output picture")
    ;
}

ProgramArgs parseArgsFromCmdLine(int argc, char** argv,
				 StatusCode& status)
{
  ProgramArgs args;

  status = StatusCode::NoError;
  std::string configFilePath;
  std::string logLevel;
  try {
    // add options for cmd line only
    po::options_description generic("Generic options");
    add_generic_options(generic, &configFilePath, &logLevel);

    // add options for both cmd line and config file
    po::options_description config("Configuration");
    add_config_options(config, &args);

    po::options_description cmdLineOptions;
    cmdLineOptions.add(generic).add(config);
    po::options_description configFileOptions;
    configFileOptions.add(config);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(cmdLineOptions).run(), vm);

    if (vm.count("help")) {
      BOOST_LOG_TRIVIAL(info) << cmdLineOptions;
      status = StatusCode::NoError;
      exit(EXIT_SUCCESS);
    }

    po::notify(vm);

    if (vm.count("log-level")) {
      bool success = boost::log::trivial::from_string(logLevel.c_str(),
						      logLevel.length(),
						      args.logLevel);
      if (!success) {
	BOOST_LOG_TRIVIAL(warning) << "Invalidy logging level. Setting to debug.";
	args.logLevel = boost::log::trivial::debug;
      }
    }

    if (!configFilePath.empty()) {
      std::ifstream ifs(configFilePath.c_str());
      if (!ifs) {
	BOOST_LOG_TRIVIAL(error) << "Cannot open config file: " << configFilePath;
	status = StatusCode::FileError;
	return args;
      }
      else {
	po::store(po::parse_config_file(ifs, configFileOptions), vm);
	po::notify(vm);
      }
    }
  }
  catch(const po::required_option& ex) {
    BOOST_LOG_TRIVIAL(error) << ex.what();
    status = StatusCode::CmdLineError;
  }
  catch(const po::unknown_option& ex) {
    BOOST_LOG_TRIVIAL(error) << ex.what();
    status = StatusCode::CmdLineError;
  }
  catch(const po::error& ex) {
    BOOST_LOG_TRIVIAL(error) << ex.what();
    status = StatusCode::CmdLineError;
  }
  return args;
}

ProgramArgs parseArgs(int argc, char** argv, StatusCode& status)
{
  return parseArgsFromCmdLine(argc, argv, status);
}
