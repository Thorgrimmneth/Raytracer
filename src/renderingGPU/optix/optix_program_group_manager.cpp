#include "optix_program_group_manager.h"

void OptixProgramGroupManager::create(OptixDeviceContext context, OptixModule raygenModule,
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

    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenPGDesc, 1, &options, nullptr, nullptr, &raygenPG));

    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = missModule;
    missPGDesc.miss.entryFunctionName = missName.c_str();

    OPTIX_CHECK(optixProgramGroupCreate(context, &missPGDesc, 1, &options, nullptr, nullptr, &missPG));

    OptixProgramGroupDesc hitPGDesc = {};
    hitPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitPGDesc.hitgroup.moduleCH = hitModule;
    hitPGDesc.hitgroup.entryFunctionNameCH = hitModule ? hitName.c_str() : nullptr;

    hitPGDesc.hitgroup.moduleAH = anyHitModule;
    hitPGDesc.hitgroup.entryFunctionNameAH = anyHitModule ? anyHitName.c_str() : nullptr;

    hitPGDesc.hitgroup.moduleIS = intersectionModule;
    hitPGDesc.hitgroup.entryFunctionNameIS = intersectionModule ? intersectionName.c_str() : nullptr;

    OPTIX_CHECK(optixProgramGroupCreate(context, &hitPGDesc, 1, &options, nullptr, nullptr, &hitPG));
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
    if (hitPG)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(hitPG));
        hitPG = nullptr;
    }
}