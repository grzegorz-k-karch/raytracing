#include <iostream>
#include <iomanip>
#include <string>

#include <cuda_runtime_api.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include "nvidia/helper_math.h"
#include "logging.h"
#include "cuda_utils.cuh"
#include "optix_utils.cuh"
#include "optix_utils.h"
#include "OptixRenderer.h"
#include "OptixRenderer.cuh"


template <typename T>
struct SbtRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


static void contextLogCB(unsigned int level,
			 const char* tag,
			 const char* message,
			 void* /*cbdata */)
{
  switch (level) {
  case 1:
    LOG_TRIVIAL(fatal) << "[OptiX]["  << tag << "]: " << message << "\n";
    break;
  case 2:
    LOG_TRIVIAL(error) << "[OptiX]["  << tag << "]: " << message << "\n";
    break;
  case 3:
    LOG_TRIVIAL(warning) << "[OptiX]["  << tag << "]: " << message << "\n";
    break;
  case 4:
    LOG_TRIVIAL(info) << "[OptiX]["  << tag << "]: " << message << "\n";
    break;
  default:
    break;
  }
}


void OptixRenderer::createContext(StatusCode& status)
{
  // Specify context options
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction       = &contextLogCB;
  options.logCallbackLevel          = 4;

  // Associate a CUDA context (and therefore a specific GPU) with this
  // device context
  CUcontext cuCtx = 0;  // zero means take the current context
  status = OCE(optixDeviceContextCreate(cuCtx, &options, &m_context));
  if (status != StatusCode::NoError) {
    return;
  }
}


void OptixRenderer::createModule(OptixPipelineCompileOptions& pipelineCompileOptions,
				 StatusCode& status)
{
  char log[2048];
  size_t sizeofLog = sizeof(log);

  OptixModuleCompileOptions moduleCompileOptions = {};
  moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

  pipelineCompileOptions.usesMotionBlur        = false;
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipelineCompileOptions.numPayloadValues      = 3;
  pipelineCompileOptions.numAttributeValues    = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
  // pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  size_t fileSize = 0;

  std::string ptxInput;
  status = loadPTXFile("objects-Release/OptixRendererPTX/src/OptixRenderer.ptx", ptxInput);
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  status = OCE(optixModuleCreateFromPTX(
					m_context,
					&moduleCompileOptions,
					&pipelineCompileOptions,
					ptxInput.c_str(),
					ptxInput.size(),
					log,
					&sizeofLog,
					&m_module
					));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
}


void OptixRenderer::createProgramGroups(StatusCode& status)
{
  char log[2048];
  size_t sizeofLog = sizeof(log);

  OptixProgramGroupOptions programGroupOptions = {};

  OptixProgramGroupDesc raygenProgGroupDesc    = {};
  raygenProgGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module            = m_module;
  raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__rg";
  status = OCE(optixProgramGroupCreate(
				       m_context,
				       &raygenProgGroupDesc,
				       1,   // num program groups
				       &programGroupOptions,
				       log,
				       &sizeofLog,
				       &m_raygenProgramGroup
				       ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  OptixProgramGroupDesc missProgGroupDesc  = {};
  missProgGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
  missProgGroupDesc.miss.module            = m_module;
  missProgGroupDesc.miss.entryFunctionName = "__miss__ms";
  status = OCE(optixProgramGroupCreate(
				       m_context,
				       &missProgGroupDesc,
				       1,   // num program groups
				       &programGroupOptions,
				       log,
				       &sizeofLog,
				       &m_missProgramGroup
				       ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  OptixProgramGroupDesc hitgroupProgGroupDesc = {};
  hitgroupProgGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroupProgGroupDesc.hitgroup.moduleCH            = m_module;
  hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  status = OCE(optixProgramGroupCreate(
				       m_context,
				       &hitgroupProgGroupDesc,
				       1,   // num program groups
				       &programGroupOptions,
				       log,
				       &sizeofLog,
				       &m_hitgroupProgramGroup
				       ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
}


void OptixRenderer::createPipeline(OptixPipelineCompileOptions& pipelineCompileOptions,
				   StatusCode& status)
{
  char log[2048];
  size_t sizeofLog = sizeof(log);

  const uint32_t maxTraceDepth = 1;
  OptixProgramGroup programGroups[] =
    {
     m_raygenProgramGroup,
     m_missProgramGroup,
     m_hitgroupProgramGroup
    };

  OptixPipelineLinkOptions pipelineLinkOptions = {};
  pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
  pipelineLinkOptions.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  status = OCE(optixPipelineCreate(
				   m_context,
				   &pipelineCompileOptions,
				   &pipelineLinkOptions,
				   programGroups,
				   sizeof(programGroups) / sizeof(programGroups[0]),
				   log,
				   &sizeofLog,
				   &m_pipeline
				   ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  OptixStackSizes stackSizes = {};
  for(auto& progGroup : programGroups) {
    status = OCE(optixUtilAccumulateStackSizes(progGroup, &stackSizes));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error\n";
    }
  }

  uint32_t directCallableStackSizeFromTraversal;
  uint32_t directCallableStackSizeFromState;
  uint32_t continuationStackSize;
  status = OCE(optixUtilComputeStackSizes(&stackSizes, maxTraceDepth,
					  0,  // maxCCDepth
					  0,  // maxDCDEpth
					  &directCallableStackSizeFromTraversal,
					  &directCallableStackSizeFromState, &continuationStackSize));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = OCE(optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal,
					 directCallableStackSizeFromState, continuationStackSize,
					 1  // maxTraversableDepth
					 ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
}


void OptixRenderer::setupShaderBindingTable(StatusCode& status)
{
  CUdeviceptr  raygenRecord;
  const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  RayGenSbtRecord rg_sbt;
  status = OCE(optixSbtRecordPackHeader(m_raygenProgramGroup, &rg_sbt));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = CCE(cudaMemcpy(
			  reinterpret_cast<void*>(raygenRecord),
			  &rg_sbt,
			  raygenRecordSize,
			  cudaMemcpyHostToDevice
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  CUdeviceptr missRecord;
  size_t      missRecordSize = sizeof(MissSbtRecord);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  MissSbtRecord ms_sbt;
  ms_sbt.data = { 0.0f, 0.0f, 0.0f };
  status = OCE(optixSbtRecordPackHeader(m_missProgramGroup, &ms_sbt));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = CCE(cudaMemcpy(
			  reinterpret_cast<void*>(missRecord),
			  &ms_sbt,
			  missRecordSize,
			  cudaMemcpyHostToDevice
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  CUdeviceptr hitgroupRecord;
  size_t      hitgroupRecordSize = sizeof(HitGroupSbtRecord);
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecord), hitgroupRecordSize));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  HitGroupSbtRecord hg_sbt;
  status = OCE(optixSbtRecordPackHeader(m_hitgroupProgramGroup, &hg_sbt));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = CCE(cudaMemcpy(
			  reinterpret_cast<void*>(hitgroupRecord),
			  &hg_sbt,
			  hitgroupRecordSize,
			  cudaMemcpyHostToDevice
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  m_shaderBindingTable = {};
  memset(&m_shaderBindingTable, 0, sizeof(OptixShaderBindingTable));
  m_shaderBindingTable.raygenRecord                = raygenRecord;
  m_shaderBindingTable.missRecordBase              = missRecord;
  m_shaderBindingTable.missRecordStrideInBytes     = sizeof(MissSbtRecord);
  m_shaderBindingTable.missRecordCount             = 1;
  m_shaderBindingTable.hitgroupRecordBase          = hitgroupRecord;
  m_shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  m_shaderBindingTable.hitgroupRecordCount         = 1;
}


OptixRenderer::OptixRenderer(StatusCode& status) :
  m_context(nullptr), m_module(nullptr),
  m_hitgroupProgramGroup(nullptr),
  m_missProgramGroup(nullptr),
  m_raygenProgramGroup(nullptr),
  m_pipeline(nullptr)
{
  // Initialize CUDA
  status = CCE(cudaFree(0));
  if (status != StatusCode::NoError) {
    return;
  }

  // Initialize the OptiX API, loading all API entry points
  status = OCE(optixInit());
  if (status != StatusCode::NoError) {
    return;
  }

  LOG_TRIVIAL(info) << "Creating Optix context...\n";
  createContext(status);

  LOG_TRIVIAL(info) << "Creating Optix module...\n";
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  createModule(pipelineCompileOptions, status);

  LOG_TRIVIAL(info) << "Creating Optix program groups...\n";
  createProgramGroups(status);

  LOG_TRIVIAL(info) << "Creating Optix pipeline...\n";
  createPipeline(pipelineCompileOptions, status);

  LOG_TRIVIAL(info) << "Setting up Optix shader binding table...\n";
  setupShaderBindingTable(status);

  LOG_TRIVIAL(info) << "Optix initialization done.\n";
}


void OptixRenderer::launch(const Camera& camera, std::vector<float3>& outputBuffer,
			   int imageWidth, int imageHeight, 
			   StatusCode& status)
{
  float3 *d_outputBuffer;
  cudaMalloc(reinterpret_cast<void**>(&d_outputBuffer), imageWidth*imageHeight*sizeof(float3));

  Camera *d_camera;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera)));
  camera.copyToDevice(d_camera, status);
  if (status != StatusCode::NoError) {
    return;
  }

  CUstream stream;
  status = CCE(cudaStreamCreate(&stream));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  Params params;
  params.image        = d_outputBuffer;
  params.image_width  = imageWidth;
  params.image_height = imageHeight;
  params.handle       = m_iasHandle;
  params.camera       = d_camera;

  CUdeviceptr d_param;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  status = CCE(cudaMemcpy(
			  reinterpret_cast<void*>(d_param),
			  &params, sizeof(params),
			  cudaMemcpyHostToDevice
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  status = OCE(optixLaunch(
			   m_pipeline,
			   stream,
			   d_param,
			   sizeof(Params),
			   &m_shaderBindingTable,
			   imageWidth,
			   imageHeight,
			   /*depth=*/1
			   )
	       );
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error in optixLaunch.\n";
  }
  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream);

  Params params2;
  status = CCE(cudaMemcpy(
  			  reinterpret_cast<void*>(&params2),
  			  reinterpret_cast<void*>(d_param), sizeof(params2),
  			  cudaMemcpyDeviceToHost
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  status = CCE(cudaMemcpy(
  			  reinterpret_cast<void*>(outputBuffer.data()),
  			  params2.image, imageWidth*imageHeight*sizeof(float3),
  			  cudaMemcpyDeviceToHost
			  ));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
}


OptixRenderer::~OptixRenderer()
{
  StatusCode status = StatusCode::NoError;
  if (m_pipeline) {
    status = OCE(optixPipelineDestroy(m_pipeline));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix pipeline destruction.";
    }
  }
  if (m_hitgroupProgramGroup) {
    status = OCE(optixProgramGroupDestroy(m_hitgroupProgramGroup));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix hit group destruction.";
    }
  }
  if (m_missProgramGroup) {
    status = OCE(optixProgramGroupDestroy(m_missProgramGroup));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix miss group destruction.";
    }
  }
  if (m_raygenProgramGroup) {
    status = OCE(optixProgramGroupDestroy(m_raygenProgramGroup));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix raygen group destruction.";
    }
  }
  if (m_module) {
    status = OCE(optixModuleDestroy(m_module));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix module destruction.";
    }
  }
  if (m_context) {
    status = OCE(optixDeviceContextDestroy(m_context));
    if (status != StatusCode::NoError) {
      LOG_TRIVIAL(error) << "Error occurred during Optix context destruction.";
    }
  }
}


void OptixRenderer::buildRootAccelStruct(std::vector<OptixTraversableHandle>& traversableHandles,
					 StatusCode& status)
{
  int numInstances = traversableHandles.size();
  CUdeviceptr d_instances;
  size_t instanceSizeInBytes = sizeof(OptixInstance)*numInstances;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_instances), instanceSizeInBytes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  OptixBuildInput instanceInput = {};
  instanceInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instanceInput.instanceArray.instances    = d_instances;
  instanceInput.instanceArray.numInstances = numInstances;

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes iasBufferSizes;
  status = OCE(optixAccelComputeMemoryUsage(m_context, &accelOptions, &instanceInput,
  					    1,  // num build inputs
  					    &iasBufferSizes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  CUdeviceptr d_tempBuffer;
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), iasBufferSizes.tempSizeInBytes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }
  status = CCE(cudaMalloc(reinterpret_cast<void**>(&m_d_iasOutputBuffer), iasBufferSizes.outputSizeInBytes));
  if (status != StatusCode::NoError) {
    LOG_TRIVIAL(error) << "Error\n";
  }

  // Use the identity matrix for the instance transform
  float transform[12] = {1,0,0,0,
  			 0,1,0,0,
  			 0,0,1,0};
  std::vector<OptixInstance> instances;
  instances.resize(numInstances);
  memset(instances.data(), 0, instanceSizeInBytes);

  int instanceId = 0;
  for (auto &instance : instances) {
    instance.traversableHandle = traversableHandles[instanceId];
    instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
    instance.instanceId        = instanceId;
    instance.sbtOffset         = 0;
    instance.visibilityMask    = 1;
    memcpy(instance.transform, transform, sizeof(float)*12);
    instanceId++;
  }

  CCE(cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), instanceSizeInBytes,
		 cudaMemcpyHostToDevice));

  OCE(optixAccelBuild(m_context,
		      0,  // CUDA stream
		      &accelOptions,
		      &instanceInput,
		      1,  // num build inputs
		      d_tempBuffer,
		      iasBufferSizes.tempSizeInBytes,
		      m_d_iasOutputBuffer,
		      iasBufferSizes.outputSizeInBytes,
		      &m_iasHandle,
		      nullptr,  // emitted property list
		      0         // num emitted properties
		      ));

  CCE(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
  CCE(cudaFree(reinterpret_cast<void*>(d_instances)));
}
