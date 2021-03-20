#include "logging.h"
#include "cuda_utils.cuh"

void checkCuda(cudaError_t result, char const *const func,
	       const char *const file, int const line)
{
  if (result) {
    LOG_TRIVIAL(error) << "CUDA error = "
		       << static_cast<unsigned int>(result)
		       << " : " << cudaGetErrorString(result) << " at "
		       << file << ":" << line << " '" << func << "'" << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}
