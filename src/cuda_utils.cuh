#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "StatusCode.h"

#define CCE(val) checkCuda( (val), #val, __FILE__, __LINE__ )
StatusCode checkCuda(cudaError_t result,
		      char const *const func,
		      const char *const file,
		      int const line);

#endif//CUDA_UTILS_CUH
