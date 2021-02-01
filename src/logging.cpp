#include <iostream>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include "Logger.h"

namespace logging = boost::log;
namespace keywords = boost::log::keywords;

void initLogger()
{
  logging::register_simple_formatter_factory<
    logging::trivial::severity_level, char>("Severity");

  logging::add_file_log
    (
     keywords::file_name = "sample.log",
     keywords::format = "[%TimeStamp%] [%ThreadID%] [%ProcessID%] [%LineID%] [%Severity%] %Message%"
     );

  logging::add_console_log
    (
     std::cout, keywords::format = "[%TimeStamp%] [%ThreadID%] [%ProcessID%] [%LineID%] [%Severity%] %Message%"
     );

  logging::core::get()->set_filter
    (
     logging::trivial::severity >= logging::trivial::info
     );

  logging::add_common_attributes();
}
