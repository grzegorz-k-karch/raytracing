#include <cstdlib>
#include <iostream>

#include "StatusCodes.h"

void exitIfError(const StatusCodes& status)
{
  if (status != StatusCodes::NoError) {
    std::cerr << "Error: "
	      << static_cast<std::underlying_type<StatusCodes>::type>(status)
	      << std::endl << "Exiting." << std::endl;
    exit(EXIT_FAILURE);
  }
}
