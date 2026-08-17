#include "optix_pipeline_manager.h"

#include <vector>

#include "../utils/simplified_def.cuh"

void OptixPipelineManager::create(OptixDeviceContext context, const OptixPipelineCompileOptions &pipelineCompileOptions,
                                  const OptixProgramGroupManager &programGroups)
{
    std::vector<OptixProgramGroup> groups;

    groups.reserve(1 + programGroups.missPGs.size() + programGroups.meshHitPGs.size() + programGroups.sdfHitPGs.size());

    // Raygen
    groups.push_back(programGroups.raygenPG);

    // Miss
    for (OptixProgramGroup pg : programGroups.missPGs)
        groups.push_back(pg);

    // Mesh hitgroups
    for (OptixProgramGroup pg : programGroups.meshHitPGs)
        groups.push_back(pg);

    // SDF hitgroups
    for (OptixProgramGroup pg : programGroups.sdfHitPGs)
        groups.push_back(pg);

    OptixPipelineLinkOptions linkOptions = {};

    linkOptions.maxTraceDepth = 2;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions, &linkOptions, groups.data(),
                                    static_cast<unsigned int>(groups.size()), log, &logSize, &pipeline));

    if (logSize > 1)
    {
        std::cout << "Pipeline log:\n" << log << std::endl;
    }
}

void OptixPipelineManager::destroy()
{
    if (pipeline)
    {
        OPTIX_CHECK(optixPipelineDestroy(pipeline));

        pipeline = nullptr;
    }
}