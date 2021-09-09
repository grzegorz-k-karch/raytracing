#include <fstream>
#include <streambuf>

#include "optix_utils.h"


// https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
StatusCode loadPTXFile(const std::string ptxFilepath, std::string& ptxContent)
{
  StatusCode status = StatusCode::NoError;
  
  std::ifstream file(ptxFilepath);
  
  file.seekg(0, std::ios::end);   
  ptxContent.reserve(file.tellg());
  file.seekg(0, std::ios::beg);

  ptxContent.assign((std::istreambuf_iterator<char>(file)),
		    std::istreambuf_iterator<char>());
  
  return status;
}
