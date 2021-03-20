#include "logging.h"
#include "cuda_utils.cuh"

StatusCodes checkCuda(cudaError_t result, char const *const func,
		      const char *const file, int const line)
{
  StatusCodes status = StatusCodes::NoError;
  if (result) {
    LOG_TRIVIAL(error) << "CUDA error = "
		       << static_cast<unsigned int>(result)
		       << " : " << cudaGetErrorString(result) << " at "
		       << file << ":" << line << " '" << func << "'" << std::endl;
    cudaDeviceReset();
    status = StatusCodes::CudaError;
  }
  return status;
}
