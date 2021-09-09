#ifndef OPTIX_UTILS_CUH
#define OPTIX_UTILS_CUH

#include <optix.h>
#include "StatusCode.h"

#define OCE(val) checkOptix( (val), #val, __FILE__, __LINE__ )
StatusCode checkOptix(OptixResult result,
		      char const *const func,
		      const char *const file,
		      int const line);

#endif//OPTIX_UTILS_CUH
