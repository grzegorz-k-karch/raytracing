#include <iostream>

#include "logging.h"
#include "args_parse.h"
#include "StatusCodes.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"
#include "Renderer.cuh"
#include "ImageSaver.h"

int main(int argc, char** argv)
{
  size_t freeMemory1;
  size_t totalMemory;
  cudaMemGetInfo(&freeMemory1, &totalMemory);
  {
    StatusCodes status{StatusCodes::NoError};

    // parse command line arguments
    ProgramArgs args = parseArgs(argc, argv, status);
    exitIfError(status);

    initLogger(args.logLevel);

    // get all objects into SceneRawObjects struct
    SceneRawObjects sceneRawObjects;
    sceneRawObjects.parseScene(args.sceneFilePath, status);
    exitIfError(status);

    // construct scene on device using class hierarchy
    // for objects and materials
    SceneDevice sceneDevice;
    sceneDevice.constructScene(sceneRawObjects, status);
    exitIfError(status);

    Renderer renderer{args.imageWidth, args.imageHeight,
		      args.sampleCount, args.rayDepth};
    // initialize random state and image buffer
    renderer.initBuffers(status);
    exitIfError(status);

    // render scene
    renderer.renderScene(sceneDevice, status);
    exitIfError(status);

    std::vector<float3> image;
    renderer.getImageOnHost(image, status);
    exitIfError(status);
  
    ImageSaver imageSaver;
    // save the rendered image to file
    imageSaver.saveImage(image, args.imageWidth, args.imageHeight,
			 args.pictureFilePath, status);
    exitIfError(status);
  }
  size_t freeMemory2;
  cudaMemGetInfo(&freeMemory2, &totalMemory);
  LOG_TRIVIAL(info) << "\nfree GPU memory: " << freeMemory2/1024.0/1024.0
		    << "\ntotal GPU memory: " << totalMemory/1024.0/1024.0
		    << "\nleaked GPU memory: "
		    << (freeMemory1 - freeMemory2)/1024.0/1024.0;

  return 0;
}
