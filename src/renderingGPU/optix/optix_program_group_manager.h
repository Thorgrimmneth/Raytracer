#pragma once

#include "optix_context.h"
#include <optix.h>

class OptixProgramGroupManager
{
  public:
    void create(OptixDeviceContext context, OptixModule raygenModule, const std::string &raygenName,
                OptixModule missModule, const std::string &missName, OptixModule closestModule = nullptr,
                const std::string &closestName = "", OptixModule anyHitModule = nullptr,
                const std::string &anyHitName = "", OptixModule intersectionModule = nullptr,
                const std::string &intersectionName = "");

    void destroy();

    OptixProgramGroup raygenPG = nullptr;

    OptixProgramGroup missPG = nullptr;

    OptixProgramGroup hitPG = nullptr;
};