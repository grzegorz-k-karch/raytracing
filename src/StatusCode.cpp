#include <cstdlib>
#include <iostream>

#include "logging.h"
#include "StatusCode.h"

void exitIfError(const StatusCode& status)
{
  if (status != StatusCode::NoError) {
    BOOST_LOG_TRIVIAL(error) << "Error: "
			     << static_cast<std::underlying_type<StatusCode>::type>(status);
    BOOST_LOG_TRIVIAL(error) << "Exiting.";
    exit(EXIT_FAILURE);
  }
}

void returnIfError(const StatusCode& status)
{
  if (status != StatusCode::NoError) {
    BOOST_LOG_TRIVIAL(error) << "Error: "
			     << static_cast<std::underlying_type<StatusCode>::type>(status);
    BOOST_LOG_TRIVIAL(error) << "Exiting.";
  }
}
