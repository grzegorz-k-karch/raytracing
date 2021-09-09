#ifndef OPTIX_UTILS_H
#define OPTIX_UTILS_H

#include <string>

#include "StatusCode.h"

StatusCode loadPTXFile(const std::string ptxFilepath, std::string& ptxContent);

#endif//OPTIX_UTILS_H
