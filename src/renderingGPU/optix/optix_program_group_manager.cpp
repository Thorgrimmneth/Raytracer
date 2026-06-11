#include "optix_program_group_manager.h"

void OptixProgramGroupManager::create(
    OptixDeviceContext context,
    OptixModule raygenModule,
    OptixModule missModule,
    OptixModule hitModule
)
{
    OptixProgramGroupOptions options = {};

    OptixProgramGroupDesc raygenPGDesc = {};
    raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenPGDesc.raygen.module = raygenModule;
    raygenPGDesc.raygen.entryFunctionName = "__raygen__intersect";

    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &raygenPGDesc,
        1,
        &options,
        nullptr, nullptr,
        &raygenPG
    ));

    OptixProgramGroupDesc missPGDesc = {};
    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = missModule;
    missPGDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &missPGDesc,
        1,
        &options,
        nullptr, nullptr,
        &missPG
    ));

    OptixProgramGroupDesc hitPGDesc = {};
    hitPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitPGDesc.hitgroup.moduleCH = hitModule;
    hitPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &hitPGDesc,
        1,
        &options,
        nullptr, nullptr,
        &hitPG
    ));
}

void OptixProgramGroupManager::destroy()
{
    if (raygenPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(raygenPG));
        raygenPG = nullptr;
    }
    if (missPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(missPG));
        missPG = nullptr;
    }
    if (hitPG) {
        OPTIX_CHECK(optixProgramGroupDestroy(hitPG));
        hitPG = nullptr;
    }
}