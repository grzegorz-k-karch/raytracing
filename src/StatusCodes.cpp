#include <cstdlib>
#include <iostream>

#include "logging.h"
#include "StatusCodes.h"

void exitIfError(const StatusCodes& status)
{
  if (status != StatusCodes::NoError) {
    BOOST_LOG_TRIVIAL(error) << "Error: "
			     << static_cast<std::underlying_type<StatusCodes>::type>(status);
    BOOST_LOG_TRIVIAL(error) << "Exiting.";
    exit(EXIT_FAILURE);
  }
}
