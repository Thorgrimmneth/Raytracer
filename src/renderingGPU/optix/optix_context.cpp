#include "optix_context.h"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)
#pragma message("OPTIX_VERSION = " STRINGIFY(OPTIX_VERSION))
static void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
    std::cout << "[" << level << "] " << tag << " : " << message << std::endl;
}

void OptixContext::initialize()
{
    int cudaVersion = 0;
cuDriverGetVersion(&cudaVersion);

std::cout
    << "CUDA Driver Version = "
    << cudaVersion
    << std::endl;
    cudaFree(0);

    OPTIX_CHECK(optixInit());

    CU_CHECK(cuCtxGetCurrent(&cuContext));

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackLevel = 4;
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