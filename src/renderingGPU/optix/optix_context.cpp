#include "optix_context.h"
#include <iostream>
#include "../utils/check.cuh"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
    std::cout << "[" << level << "] " << tag << " : " << message << std::endl;
}

void OptixContext::initialize()
{
    int cudaVersion = 0;
    cuDriverGetVersion(&cudaVersion);

    std::cout << "CUDA Driver Version = " << cudaVersion << std::endl;

    pipelineCompileOptions = {};

    pipelineCompileOptions.usesMotionBlur = false;

    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

    pipelineCompileOptions.numPayloadValues = 2;

    pipelineCompileOptions.numAttributeValues = 1;

    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixInit());

    CU_CHECK(cuCtxGetCurrent(&cuContext));

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackLevel = 0; // 0 = no log, 1 = error, 2 = warning, 3 = info, 4 = debug
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &deviceContext));
}

void OptixContext::destroy()
{
    if (deviceContext)
    {
        optixDeviceContextDestroy(deviceContext);
        deviceContext = nullptr;
    }

    if (stream)
    {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}