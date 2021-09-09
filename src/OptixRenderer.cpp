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
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
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
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.numPayloadValues      = 3;
  pipelineCompileOptions.numAttributeValues    = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
  pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  size_t fileSize = 0;

  std::string ptxInput;
  status = loadPTXFile("objects-Release/OptixRendererPTX/src/OptixRenderer.ptx", ptxInput);

  // const char* input = loadCUFile("OptixRenderer.cu", fileSize);
  // size_t sizeofLog = sizeof(log);

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
}


void OptixRenderer::createProgramGroups(StatusCode& status)
{
  char log[2048];
  size_t sizeofLog = sizeof(log);

  OptixProgramGroupOptions programGroupOptions = {};

  OptixProgramGroupDesc raygenProgGroupDesc    = {};
  raygenProgGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygenProgGroupDesc.raygen.module            = m_module;
  raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__renderScene";
  OCE(optixProgramGroupCreate(
			      m_context,
			      &raygenProgGroupDesc,
			      1,   // num program groups
			      &programGroupOptions,
			      log,
			      &sizeofLog,
			      &m_raygenProgramGroup
			      ));

  OptixProgramGroupDesc missProgGroupDesc  = {};
  missProgGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
  missProgGroupDesc.miss.module            = m_module;
  missProgGroupDesc.miss.entryFunctionName = "__miss__ms";
  OCE(optixProgramGroupCreate(
			      m_context,
			      &missProgGroupDesc,
			      1,   // num program groups
			      &programGroupOptions,
			      log,
			      &sizeofLog,
			      &m_missProgramGroup
			      ));

  OptixProgramGroupDesc hitgroupProgGroupDesc = {};
  hitgroupProgGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroupProgGroupDesc.hitgroup.moduleCH            = m_module;
  hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OCE(optixProgramGroupCreate(
			      m_context,
			      &hitgroupProgGroupDesc,
			      1,   // num program groups
			      &programGroupOptions,
			      log,
			      &sizeofLog,
			      &m_hitgroupProgramGroup
			      ));
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
  OCE(optixPipelineCreate(
			  m_context,
			  &pipelineCompileOptions,
			  &pipelineLinkOptions,
			  programGroups,
			  sizeof(programGroups) / sizeof(programGroups[0]),
			  log,
			  &sizeofLog,
			  &m_pipeline
			  ));

  OptixStackSizes stackSizes = {};
  for(auto& progGroup : programGroups) {
    OCE(optixUtilAccumulateStackSizes(progGroup, &stackSizes));
  }

  uint32_t directCallableStackSizeFromTraversal;
  uint32_t directCallableStackSizeFromState;
  uint32_t continuationStackSize;
  OCE(optixUtilComputeStackSizes(&stackSizes, maxTraceDepth,
				 0,  // maxCCDepth
				 0,  // maxDCDEpth
				 &directCallableStackSizeFromTraversal,
				 &directCallableStackSizeFromState, &continuationStackSize));
  OCE(optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal,
				directCallableStackSizeFromState, continuationStackSize,
				1  // maxTraversableDepth
				));
}


void OptixRenderer::setupShaderBindingTable(StatusCode& status)
{
  CUdeviceptr  raygenRecord;
  const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
  CCE(cudaMalloc(reinterpret_cast<void**>(&raygenRecord), raygenRecordSize));
  RayGenSbtRecord rg_sbt;
  OCE(optixSbtRecordPackHeader(m_raygenProgramGroup, &rg_sbt));
  CCE(cudaMemcpy(
		 reinterpret_cast<void*>(raygenRecord),
		 &rg_sbt,
		 raygenRecordSize,
		 cudaMemcpyHostToDevice
		 ));

  CUdeviceptr missRecord;
  size_t      missRecordSize = sizeof(MissSbtRecord);
  CCE(cudaMalloc(reinterpret_cast<void**>(&missRecord), missRecordSize));
  MissSbtRecord ms_sbt;
  ms_sbt.data = { 0.3f, 0.1f, 0.2f };
  OCE(optixSbtRecordPackHeader(m_missProgramGroup, &ms_sbt));
  CCE(cudaMemcpy(
		 reinterpret_cast<void*>(missRecord),
		 &ms_sbt,
		 missRecordSize,
		 cudaMemcpyHostToDevice
		 ));

  CUdeviceptr hitgroupRecord;
  size_t      hitgroupRecordSize = sizeof(HitGroupSbtRecord);
  CCE(cudaMalloc(reinterpret_cast<void**>(&hitgroupRecord), hitgroupRecordSize));
  HitGroupSbtRecord hg_sbt;
  OCE(optixSbtRecordPackHeader(m_hitgroupProgramGroup, &hg_sbt));
  CCE(cudaMemcpy(
		 reinterpret_cast<void*>(hitgroupRecord),
		 &hg_sbt,
		 hitgroupRecordSize,
		 cudaMemcpyHostToDevice
		 ));

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


void OptixRenderer::launch(OptixTraversableHandle gasHandle,
			   StatusCode& status)
{
  unsigned int imageWidth = 1200;
  unsigned int imageHeight = 800;

  uchar4* d_outputBuffer;
  cudaMalloc(reinterpret_cast<void**>(d_outputBuffer), imageWidth*imageHeight*sizeof(uchar4));

  CUstream stream;
  CCE(cudaStreamCreate(&stream));

  // sutil::Camera cam;
  // configureCamera(cam, width, height);

  Params params;
  params.image        = d_outputBuffer;
  params.image_width  = imageWidth;
  params.image_height = imageHeight;
  params.handle       = gasHandle;
  params.cam_eye      = make_float3(0.0f, 0.0f, 2.0f);
  params.cam_u        = make_float3(1.0f, 0.0f, 0.0f);
  params.cam_v        = make_float3(0.0f, 1.0f, 0.0f);
  params.cam_w        = make_float3(0.0f, 0.0f, 1.0f);

  CUdeviceptr d_param;
  CCE(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
  CCE(cudaMemcpy(
		 reinterpret_cast<void*>(d_param),
		 &params, sizeof(params),
		 cudaMemcpyHostToDevice
		 ));

  OCE(optixLaunch(
		  m_pipeline,
		  stream,
		  d_param,
		  sizeof(Params),
		  &m_shaderBindingTable,
		  imageWidth,
		  imageHeight,
		  /*depth=*/1));
  cudaDeviceSynchronize();
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
