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
  ProgramArgs programArgs;
  status = parseArgs(argc, argv, programArgs);
  exitIfError(status);

  initLogger(programArgs.logLevel);

  OptixRenderer optixRenderer(status);
  exitIfError(status);

  // get all objects into SceneRawObjects struct
  SceneRawObjects sceneRawObjects;
  status = sceneRawObjects.loadScene(programArgs.sceneFilePath);
  exitIfError(status);

  OptixTraversableHandle traversableHandle;
  sceneRawObjects.generateTraversableHandle(optixRenderer.getContext(),
					    &traversableHandle);
  std::vector<HitGroupSBTRecord> hitgroupRecords;
  sceneRawObjects.generateHitGroupRecords(hitgroupRecords);
  status = optixRenderer.setupShaderBindingTable(hitgroupRecords);
  
  status = optixRenderer.buildRootAccelStruct(traversableHandle);

  Camera camera = sceneRawObjects.getCamera();
  std::vector<float3> image;
  image.resize(programArgs.imageWidth*programArgs.imageHeight);
  status = optixRenderer.launch(camera,
  				image,
  				programArgs.imageWidth,
  				programArgs.imageHeight);

  ImageSaver imageSaver;
  // save the rendered image to file
  imageSaver.saveImage(image, programArgs.imageWidth, programArgs.imageHeight,
  		       programArgs.pictureFilePath, status);
  exitIfError(status);

  return 0;
}
