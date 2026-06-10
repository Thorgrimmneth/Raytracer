#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <iostream>
#include "../utils/macro.cuh"

class OptixContext
{
public:

    void initialize();
    void destroy();

    CUcontext           cuContext = nullptr;
    CUstream            stream = nullptr;
    OptixDeviceContext  deviceContext = nullptr;
};