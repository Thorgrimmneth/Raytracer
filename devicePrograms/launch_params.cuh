#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../src/renderingGPU/utils/objectType.h"
#include "../src/renderingGPU/raytracingUtils/ray.cuh"
#include "../src/renderingGPU/utils/op.cuh"

struct OptixHit
{
    float4 position;
    float3 normal;

    float t;

    int materialIndex;
    int objectIndex;
    HitObjectType objectType;

    D_FORCEINLINE void setHitInfo(const float4 &p, const float3 &n, float distance, int matIndex, int objIndex,
                                  HitObjectType objType)
    {
        position = p;
        t = distance;
        normal = n;
        materialIndex = matIndex;
        objectIndex = objIndex;
        objectType = objType;
    }
    

    D_FORCEINLINE void faceNormal(const float4 &direction) { normal = dot(direction, normal) < 0.f ? normal : -normal; }
};

struct LaunchParams
{
    float4* origins = nullptr;

    float4* directions = nullptr;

    OptixHit* hits = nullptr;

    int* hitMask = nullptr;

    int* activeQueue = nullptr;

    int activeCount = 0;

    OptixTraversableHandle traversable = 0;
};