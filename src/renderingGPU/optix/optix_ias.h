#pragma once

#include "optix_gas.h"
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>

class OptixIAS
{
  public:
    void build(OptixDeviceContext context, const std::vector<OptixInstance> &instances);

    OptixTraversableHandle handle;

    inline CUdeviceptr getBuffer() { return d_buffer; }
    inline CUdeviceptr getInstances() { return d_instances; }

  private:
    CUdeviceptr d_instances;
    CUdeviceptr d_buffer;
};