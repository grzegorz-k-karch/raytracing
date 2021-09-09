#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "logging.h"
#include "optix_utils.cuh"

StatusCode checkOptix(OptixResult result, char const *const func,
		      const char *const file, int const line)
{
  StatusCode status = StatusCode::NoError;
  if (result != OPTIX_SUCCESS) {
    LOG_TRIVIAL(error) << "OPTIX error = "
		       << static_cast<unsigned int>(result)
		       << " : " << optixGetErrorString(result) << " at "
		       << file << ":" << line << " '" << func
		       << "'" << std::endl;
    status = StatusCode::OptixError;
  }
  return status;
}
