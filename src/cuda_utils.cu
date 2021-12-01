#include "logging.h"
#include "cuda_utils.cuh"

StatusCode checkCuda(cudaError_t result, char const *const func,
		     const char *const file, int const line)
{
  StatusCode status = StatusCode::NoError;
  if (result) {
    LOG_TRIVIAL(error) << "CUDA error = "
		       << static_cast<unsigned int>(result)
		       << " : " << cudaGetErrorString(result) << " at "
		       << file << ":" << line << " '" << func << "'" << std::endl;
    cudaDeviceReset();
    status = StatusCode::CudaError;
  }
  return status;
}
