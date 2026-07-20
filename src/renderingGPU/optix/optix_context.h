#pragma once

#include <cuda_runtime.h>
#include <optix_stubs.h>

class OptixContext
{
public:

    void initialize();
    void destroy();

    CUcontext           cuContext = nullptr;
    CUstream            stream = nullptr;
    OptixDeviceContext  deviceContext = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
};