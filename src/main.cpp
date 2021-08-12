#include <iostream>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "logging.h"
#include "args_parse.h"
#include "StatusCode.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"
#include "Renderer.cuh"
#include "ImageSaver.h"

static void context_log_cb(unsigned int level,
			   const char* tag,
			   const char* message,
			   void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


int main(int argc, char** argv)
{
  StatusCode status{StatusCode::NoError};

  // parse command line arguments
  ProgramArgs args = parseArgs(argc, argv, status);
  exitIfError(status);

  initLogger(args.logLevel);

  // TODO: add error checking
  OptixDeviceContext context = nullptr;
  {
    // Initialize CUDA
    cudaFree(0);
    // Initialize the OptiX API, loading all API entry points
    optixInit();
    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0;  // zero means take the current context
    optixDeviceContextCreate(cuCtx, &options, &context);
  }

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

  return 0;
}
