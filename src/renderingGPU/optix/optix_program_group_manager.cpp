#include "optix_program_group_manager.h"

void OptixProgramGroupManager::create(OptixContext context, OptixModule raygenModule,
                                      const std::string &raygenName, OptixModule missModule,
                                      const std::string &missName, OptixModule hitModule, const std::string &hitName,
                                      OptixModule anyHitModule, const std::string &anyHitName, OptixModule intersectionModule, const std::string &intersectionName)
{
    if (!hitModule && !anyHitModule)
    {
        throw std::runtime_error("HitGroup must contain a ClosestHit or an AnyHit program.");
    }

    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenPGDesc.raygen.module = raygenModule;
    raygenPGDesc.raygen.entryFunctionName = raygenName.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &raygenPGDesc, 1, &options, nullptr, nullptr, &raygenPG));

    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = missModule;
    missPGDesc.miss.entryFunctionName = missName.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &missPGDesc, 1, &options, nullptr, nullptr, &missPG));

    OptixProgramGroupDesc hitPGDesc = {};
    hitPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitPGDesc.hitgroup.moduleCH = hitModule;
    hitPGDesc.hitgroup.entryFunctionNameCH = hitModule ? hitName.c_str() : nullptr;

    hitPGDesc.hitgroup.moduleAH = anyHitModule;
    hitPGDesc.hitgroup.entryFunctionNameAH = anyHitModule ? anyHitName.c_str() : nullptr;

    hitPGDesc.hitgroup.moduleIS = intersectionModule;
    hitPGDesc.hitgroup.entryFunctionNameIS = intersectionModule ? intersectionName.c_str() : nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &hitPGDesc, 1, &options, nullptr, nullptr, &meshHitPG));
}

void OptixProgramGroupManager::addRaygenProgram(OptixContext context, const std::string &raygenModulePath, const std::string &raygenName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    OptixModuleManager raygenModule;
    raygenModule.createFromPath(context, raygenModulePath);
    modules.push_back(raygenModule);
    raygenPGDesc.raygen.module = raygenModule.module;
    raygenPGDesc.raygen.entryFunctionName = raygenName.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &raygenPGDesc, 1, &options, nullptr, nullptr, &raygenPG));
}

void OptixProgramGroupManager::addMissProgram(OptixContext context, const std::string &missModulePath, const std::string &missName)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    OptixModuleManager missModule;
    missModule.createFromPath(context, missModulePath);
    modules.push_back(missModule);
    missPGDesc.miss.module = missModule.module;
    missPGDesc.miss.entryFunctionName = missName.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &missPGDesc, 1, &options, nullptr, nullptr, &missPG));
}

void OptixProgramGroupManager::addMeshHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                                                  const std::string &anyHitModulePath, const std::string &anyHitName,
                                                  const std::string &intersectionModulePath, const std::string &intersectionName)
{
    if (closestModulePath.empty() && anyHitModulePath.empty())
    {
        throw std::runtime_error("HitGroup must contain a ClosestHit or an AnyHit program.");
    }

    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc hitPGDesc = {};
    hitPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixModuleManager closestModule;
    if(!closestModulePath.empty()) 
    {
        closestModule.createFromPath(context, closestModulePath);
        modules.push_back(closestModule);
    }
    hitPGDesc.hitgroup.moduleCH = closestModule.module;
    hitPGDesc.hitgroup.entryFunctionNameCH = closestModule.module ? closestName.c_str() : nullptr;

    OptixModuleManager anyHitModule;
    if (!anyHitModulePath.empty()) 
    {
        anyHitModule.createFromPath(context, anyHitModulePath);
        modules.push_back(anyHitModule);
    }
    hitPGDesc.hitgroup.moduleAH = anyHitModule.module;
    hitPGDesc.hitgroup.entryFunctionNameAH = anyHitModule.module ? anyHitName.c_str() : nullptr;

    OptixModuleManager intersectionModule;
    if (!intersectionModulePath.empty()) 
    {
        intersectionModule.createFromPath(context, intersectionModulePath);
        modules.push_back(intersectionModule);
    }
    hitPGDesc.hitgroup.moduleIS = intersectionModule.module;
    hitPGDesc.hitgroup.entryFunctionNameIS = intersectionModule.module ? intersectionName.c_str() : nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &hitPGDesc, 1, &options, nullptr, nullptr, &meshHitPG));
}

void OptixProgramGroupManager::addSdfHitProgram(OptixContext context, const std::string &closestModulePath, const std::string &closestName,
                                                const std::string &anyHitModulePath, const std::string &anyHitName,
                                                const std::string &intersectionModulePath, const std::string &intersectionName)
{
    if (closestModulePath.empty() && anyHitModulePath.empty())
    {
        throw std::runtime_error("HitGroup must contain a ClosestHit or an AnyHit program.");
    }

    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc hitPGDesc = {};
    hitPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixModuleManager closestModule;
    if (!closestModulePath.empty()) 
    {
        closestModule.createFromPath(context, closestModulePath);
        modules.push_back(closestModule);
    }
    hitPGDesc.hitgroup.moduleCH = closestModule.module;
    hitPGDesc.hitgroup.entryFunctionNameCH = closestModule.module ? closestName.c_str() : nullptr;

    OptixModuleManager anyHitModule;
    if (!anyHitModulePath.empty()) {
        anyHitModule.createFromPath(context, anyHitModulePath);
        modules.push_back(anyHitModule);
    }
    hitPGDesc.hitgroup.moduleAH = anyHitModule.module;
    hitPGDesc.hitgroup.entryFunctionNameAH = anyHitModule.module ? anyHitName.c_str() : nullptr;

    OptixModuleManager intersectionModule;
    if (!intersectionModulePath.empty()) {
        intersectionModule.createFromPath(context, intersectionModulePath);
        modules.push_back(intersectionModule);
    }
    hitPGDesc.hitgroup.moduleIS = intersectionModule.module;
    hitPGDesc.hitgroup.entryFunctionNameIS = intersectionModule.module ? intersectionName.c_str() : nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context.deviceContext, &hitPGDesc, 1, &options, nullptr, nullptr, &sdfHitPG));
}

void OptixProgramGroupManager::destroy()
{
    if (raygenPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(raygenPG));
        raygenPG = nullptr;
    }
    if (missPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(missPG));
        missPG = nullptr;
    }
    if (meshHitPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(meshHitPG));
        meshHitPG = nullptr;
    }
    if( sdfHitPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(sdfHitPG));
        sdfHitPG = nullptr;
    }
}