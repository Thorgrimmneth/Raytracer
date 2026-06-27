#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <string>

#include "../utils/check.cuh"

class OptixModuleManager
{
  public:
    void create(OptixDeviceContext context, const std::string &ptx);
    void createFromPath(OptixDeviceContext context, const std::string &path);
    void destroy();

    OptixModule module = nullptr;

      const OptixPipelineCompileOptions& getPipelineCompileOptions() const
      {
          return pipelineCompileOptions;
      }
  private:
    OptixPipelineCompileOptions pipelineCompileOptions = {};
};