#pragma once

#include "optix_context.h"
#include <optix.h>
#include <vector>
#include "optix_module_manager.h"

class OptixProgramGroupManager
{
  public:
    void create(OptixContext context, OptixModule raygenModule, const std::string &raygenName,
                OptixModule missModule, const std::string &missName, OptixModule closestModule = nullptr,
                const std::string &closestName = "", OptixModule anyHitModule = nullptr,
                const std::string &anyHitName = "", OptixModule intersectionModule = nullptr,
                const std::string &intersectionName = "");

    void addRaygenProgram(OptixContext context, const std::string &raygenModulePath, const std::string &raygenName);

    void addMissProgram(OptixContext context, const std::string &missModulePath, const std::string &missName);

    void addMeshHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                           const std::string &anyHitModulePath = "", const std::string &anyHitName = "",
                           const std::string &intersectionModulePath = "", const std::string &intersectionName = "");

    void addSdfHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                          const std::string &anyHitModulePath = "", const std::string &anyHitName = "",
                          const std::string &intersectionModulePath = "", const std::string &intersectionName = "");
    void destroy();

    OptixProgramGroup raygenPG = nullptr;

    OptixProgramGroup missPG = nullptr;

    OptixProgramGroup meshHitPG = nullptr;

    OptixProgramGroup sdfHitPG = nullptr;

    std::vector<OptixModuleManager> modules; // Store module managers to keep them alive
};