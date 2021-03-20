#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#define CCE(val) checkCuda( (val), #val, __FILE__, __LINE__ )
void checkCuda(cudaError_t result,
	       char const *const func,
	       const char *const file,
	       int const line);

#endif//CUDA_UTILS_CUH
