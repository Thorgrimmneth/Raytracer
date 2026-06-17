#pragma once

#include "optix_context.h"
#include <optix.h>

class OptixProgramGroupManager
{
  public:
    void create(OptixDeviceContext context, OptixModule raygenModule, OptixModule missModule, OptixModule hitModule);

    void destroy();

    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitPG = nullptr;
};