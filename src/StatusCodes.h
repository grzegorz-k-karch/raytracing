#ifndef STATUS_CODES_H
#define STATUS_CODES_H

enum class StatusCodes {
  NoError, FileError, CmdLineError, SceneError, UnknownError, CudaError
    };

// used in main
void exitIfError(const StatusCodes& status);

// used in any other function
void returnIfError(const StatusCodes& status);

#endif//STATUS_CODES_H
