#include "gkk_cuda_utils.cuh"

#include <iostream>

void check_cuda(cudaError_t result,
		char const *const func,
		const char *const file,
		int const line)
{
  if (result) {
    std::cerr << "CUDA error = "
      << static_cast<unsigned int>(result)
	      << " : " << cudaGetErrorString(result) << " at "
	      << file << ":" << line << " '" << func << "'" << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}
