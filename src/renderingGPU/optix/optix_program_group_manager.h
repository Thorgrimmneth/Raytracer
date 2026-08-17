#pragma once

#include "optix_context.h"
#include <optix.h>
#include <vector>

#include "optix_module_manager.h"
#include "optix_ray_type.h"

class OptixProgramGroupManager
{
  public:
    void addRaygenProgram(OptixContext context, const std::string &raygenModulePath, const std::string &raygenName);

    void addMissProgram(OptixContext context, const std::string &missModulePath, const std::string &missName);

    void addMeshHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                           const std::string &anyHitModulePath = "", const std::string &anyHitName = "",
                           const std::string &intersectionModulePath = "", const std::string &intersectionName = "");

    void addSdfHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                          const std::string &anyHitModulePath = "", const std::string &anyHitName = "",
                          const std::string &intersectionModulePath = "", const std::string &intersectionName = "");

    void destroy();

    // --------------------------------------------------
    // Program groups
    // --------------------------------------------------

    OptixProgramGroup raygenPG = nullptr;

    std::vector<OptixProgramGroup> missPGs;

    std::vector<OptixProgramGroup> meshHitPGs;

    std::vector<OptixProgramGroup> sdfHitPGs;

    // --------------------------------------------------
    // Modules
    // --------------------------------------------------

    std::vector<OptixModuleManager> modules;
};