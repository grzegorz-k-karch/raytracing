#include <iostream>
#include <optix.h>

#include "logging.h"
#include "args_parse.h"
#include "StatusCode.h"
#include "SceneRawObjects.h"
#include "SceneDevice.cuh"
#include "Renderer.cuh"
#include "ImageSaver.h"
#include "OptixRenderer.h"


int main(int argc, char** argv)
{
  StatusCode status = StatusCode::NoError;

  // parse command line arguments
  ProgramArgs args = parseArgs(argc, argv, status);
  exitIfError(status);

  initLogger(args.logLevel);

  OptixRenderer optixRenderer(status);

  // get all objects into SceneRawObjects struct
  SceneRawObjects sceneRawObjects;
  sceneRawObjects.parseScene(args.sceneFilePath, status);
  exitIfError(status);

  sceneRawObjects.copyToDevice(status);
  exitIfError(status);
  
  std::vector<OptixTraversableHandle> traversableHandles;
  sceneRawObjects.generateTraversableHandles(optixRenderer.getContext(),
					     traversableHandles);
  optixRenderer.buildRootAccelStruct(traversableHandles,
				     status);

  Camera camera = sceneRawObjects.getCamera();
  std::vector<float3> image;
  image.resize(args.imageWidth*args.imageHeight);
  optixRenderer.launch(camera, image, status);

  ImageSaver imageSaver;
  // save the rendered image to file
  imageSaver.saveImage(image, args.imageWidth, args.imageHeight,
  		       args.pictureFilePath, status);
  exitIfError(status);

  return 0;
}
