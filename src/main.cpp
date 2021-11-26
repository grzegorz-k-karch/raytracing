#include <iostream>
#include <optix.h>

#include "logging.h"
#include "args_parse.h"
#include "StatusCode.h"
#include "SceneRawObjects.h"
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
  exitIfError(status);

  // get all objects into SceneRawObjects struct
  SceneRawObjects sceneRawObjects;
  sceneRawObjects.loadScene(args.sceneFilePath, status);
  exitIfError(status);

  std::vector<OptixTraversableHandle> traversableHandles;
  sceneRawObjects.generateTraversableHandles(optixRenderer.getContext(),
  					     traversableHandles);
  optixRenderer.buildRootAccelStruct(traversableHandles,
  				     status);

  Camera camera = sceneRawObjects.getCamera();
  std::vector<float3> image;
  image.resize(args.imageWidth*args.imageHeight);
  optixRenderer.launch(camera,
  		       image,
  		       args.imageWidth,
  		       args.imageHeight,
  		       status);

  ImageSaver imageSaver;
  // save the rendered image to file
  imageSaver.saveImage(image, args.imageWidth, args.imageHeight,
  		       args.pictureFilePath, status);
  exitIfError(status);

  return 0;
}
