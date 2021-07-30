#ifndef STATUS_CODES_H
#define STATUS_CODES_H

enum class StatusCode {
  NoError, FileError, CmdLineError, SceneError, UnknownError, CudaError
    };

// used in main
void exitIfError(const StatusCode& status);

// used in any other function
void returnIfError(const StatusCode& status);

#endif//STATUS_CODES_H
