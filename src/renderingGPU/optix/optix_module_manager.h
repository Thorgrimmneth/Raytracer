#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <string>
#include "optix_context.h"
#include "../utils/check.cuh"

class OptixModuleManager
{
  public:
    void create(OptixContext context, const std::string &ptx);
    void createFromPath(OptixContext context, const std::string &path);
    void destroy();

    OptixModule module = nullptr;

      const OptixPipelineCompileOptions& getPipelineCompileOptions() const
      {
          return pipelineCompileOptions;
      }
  private:
    OptixPipelineCompileOptions pipelineCompileOptions = {};
};