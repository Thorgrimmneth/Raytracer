#include "optix_program_group_manager.h"

void OptixProgramGroupManager::addRaygenProgram(OptixContext context, const std::string &raygenModulePath,
                                                const std::string &raygenName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    OptixModuleManager raygenModule;
    raygenModule.createFromPath(context, raygenModulePath);
    modules.push_back(raygenModule);
    raygenPGDesc.raygen.module = raygenModule.module;
    raygenPGDesc.raygen.entryFunctionName = raygenName.c_str();

    OPTIX_CHECK(
        optixProgramGroupCreate(context.deviceContext, &raygenPGDesc, 1, &options, nullptr, nullptr, &raygenPG));
}

void OptixProgramGroupManager::addMissProgram(OptixContext context, const std::string &missModulePath,
                                              const std::string &missName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    OptixModuleManager module;
    module.createFromPath(context, missModulePath);

    modules.push_back(module);

    desc.miss.module = module.module;
    desc.miss.entryFunctionName = missName.c_str();

    OptixProgramGroup pg = nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &desc, 1, &options, nullptr, nullptr, &pg));

    missPGs.push_back(pg);
}

void OptixProgramGroupManager::addMeshHitProgram(OptixContext context, const std::string &closestModulePath,
                                                 const std::string &closestName, const std::string &anyHitModulePath,
                                                 const std::string &anyHitName,
                                                 const std::string &intersectionModulePath,
                                                 const std::string &intersectionName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    OptixModuleManager closestModule;
    OptixModuleManager anyHitModule;
    OptixModuleManager intersectionModule;

    if (!closestModulePath.empty())
    {
        closestModule.createFromPath(context, closestModulePath);
        modules.push_back(closestModule);

        desc.hitgroup.moduleCH = closestModule.module;
        desc.hitgroup.entryFunctionNameCH = closestName.c_str();
    }

    if (!anyHitModulePath.empty())
    {
        anyHitModule.createFromPath(context, anyHitModulePath);
        modules.push_back(anyHitModule);

        desc.hitgroup.moduleAH = anyHitModule.module;
        desc.hitgroup.entryFunctionNameAH = anyHitName.c_str();
    }

    if (!intersectionModulePath.empty())
    {
        intersectionModule.createFromPath(context, intersectionModulePath);
        modules.push_back(intersectionModule);

        desc.hitgroup.moduleIS = intersectionModule.module;
        desc.hitgroup.entryFunctionNameIS = intersectionName.c_str();
    }

    OptixProgramGroup pg = nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &desc, 1, &options, nullptr, nullptr, &pg));

    meshHitPGs.push_back(pg);
}

void OptixProgramGroupManager::addSdfHitProgram(OptixContext context, const std::string &closestModulePath,
                                                const std::string &closestName, const std::string &anyHitModulePath,
                                                const std::string &anyHitName,
                                                const std::string &intersectionModulePath,
                                                const std::string &intersectionName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    OptixModuleManager closestModule;
    OptixModuleManager anyHitModule;
    OptixModuleManager intersectionModule;

    if (!closestModulePath.empty())
    {
        closestModule.createFromPath(context, closestModulePath);
        modules.push_back(closestModule);

        desc.hitgroup.moduleCH = closestModule.module;
        desc.hitgroup.entryFunctionNameCH = closestName.c_str();
    }

    if (!anyHitModulePath.empty())
    {
        anyHitModule.createFromPath(context, anyHitModulePath);
        modules.push_back(anyHitModule);

        desc.hitgroup.moduleAH = anyHitModule.module;
        desc.hitgroup.entryFunctionNameAH = anyHitName.c_str();
    }

    if (!intersectionModulePath.empty())
    {
        intersectionModule.createFromPath(context, intersectionModulePath);
        modules.push_back(intersectionModule);

        desc.hitgroup.moduleIS = intersectionModule.module;
        desc.hitgroup.entryFunctionNameIS = intersectionName.c_str();
    }

    OptixProgramGroup pg = nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &desc, 1, &options, nullptr, nullptr, &pg));

    sdfHitPGs.push_back(pg);
}

void OptixProgramGroupManager::destroy()
{
    // Raygen
    if (raygenPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(raygenPG));
        raygenPG = nullptr;
    }

    // Miss programs
    for (OptixProgramGroup pg : missPGs)
    {
        if (pg)
        {
            OPTIX_CHECK(optixProgramGroupDestroy(pg));
        }
    }
    missPGs.clear();

    // Mesh hit groups
    for (OptixProgramGroup pg : meshHitPGs)
    {
        if (pg)
        {
            OPTIX_CHECK(optixProgramGroupDestroy(pg));
        }
    }
    meshHitPGs.clear();

    // SDF hit groups
    for (OptixProgramGroup pg : sdfHitPGs)
    {
        if (pg)
        {
            OPTIX_CHECK(optixProgramGroupDestroy(pg));
        }
    }
    sdfHitPGs.clear();

    // Modules
    for (auto &module : modules)
    {
        module.destroy();
    }
    modules.clear();
}