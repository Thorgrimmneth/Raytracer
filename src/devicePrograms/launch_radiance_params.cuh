#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../renderingGPU/utils/rng.cuh"
#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/materials/material.cuh"
#include "../renderingGPU/raytracingUtils/ray.cuh"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/objects/triangle_mesh.cuh"
#include "../renderingGPU/utils/shading_data.cuh"
#include "../renderingGPU/lights/light_context.cuh"
#include "../renderingGPU/lights/light.cuh"

struct LaunchRadianceParams
{
    float3* origins;
    float3* directions;
    float3* accum_buffer;
    float3* throughputs;
    int* lastBounceWasDelta;  // Tracks if previous bounce was from delta material
    int* isInside;
    RNG* rngs;

    int active_count;
    int depth;
    OptixTraversableHandle traversable;

    int nbMeshInstances;
    int nbMaterials;
    int maxBounces;
    float3 sunDirection;

    float HR;
    float HM;

    float3 betaR;
    float3 betaM;

    float atmosphereSize;

    int nbSkySamples;

    float sunAngularRadius;
    float sunHalfAngularRadius;

    int nbLights;
    Light* lights;
    float* lightCumulativeWeights;
    float* lightProbabilities;
    LightContext lightContext;
};