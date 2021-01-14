#ifndef STATUS_CODES_H
#define STATUS_CODES_H

enum class StatusCodes {
  NoError, FileError, CmdLineError, SceneError, UnknownError
    };

void exitIfError(const StatusCodes& status);

#endif//STATUS_CODES_H
