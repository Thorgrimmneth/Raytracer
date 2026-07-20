#include "optix_pipeline_manager.h"

#include <vector>

#include "../utils/simplified_def.cuh"


void OptixPipelineManager::create(
    OptixDeviceContext context,
    const OptixPipelineCompileOptions& pipelineCompileOptions,
    const OptixProgramGroupManager& programGroups
)
{
    //TODO : add sdf module to the pipeline creation
    std::vector<OptixProgramGroup> groups =
    {
        programGroups.raygenPG,
        programGroups.missPG,
        programGroups.meshHitPG,
        programGroups.sdfHitPG
    };

    OptixPipelineLinkOptions linkOptions = {};

    linkOptions.maxTraceDepth = 1;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(
        optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &linkOptions,
            groups.data(),
            static_cast<unsigned int>(groups.size()),
            log,
            &logSize,
            &pipeline
        )
    );

    if(logSize > 1)
    {
        std::cout
            << "Pipeline log:\n"
            << log
            << std::endl;
    }
}

void OptixPipelineManager::create(
    OptixDeviceContext context,
    const OptixPipelineCompileOptions& pipelineCompileOptions,
    const std::vector<OptixProgramGroup>& programGroups
)
{
    OptixPipelineLinkOptions linkOptions = {};

    linkOptions.maxTraceDepth = 1;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(
        optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &linkOptions,
            programGroups.data(),
            static_cast<unsigned int>(programGroups.size()),
            log,
            &logSize,
            &pipeline
        )
    );

    if(logSize > 1)
    {
        std::cout
            << "Pipeline log:\n"
            << log
            << std::endl;
    }
}

void OptixPipelineManager::destroy()
{
    if(pipeline)
    {
        OPTIX_CHECK(
            optixPipelineDestroy(
                pipeline
            )
        );

        pipeline = nullptr;
    }
}