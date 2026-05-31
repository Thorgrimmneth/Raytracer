#pragma once

#include "../utils/rng.cuh"
#include "../utils/macro.cuh"

#include "../materials/material.cuh"

#include "../objects/triangle_mesh_geometry.cuh"
#include "../objectsUtils/aabb.cuh"

#include "../raytracingUtils/hitrecord.cuh"
#include "../raytracingUtils/ray.cuh"


enum WavefrontQueueType
{
    QUEUE_LAMBERT = 0,
    QUEUE_EMISSIVE = 1,
    QUEUE_METAL = 2,
    QUEUE_MIRROR = 3,
    QUEUE_PLASTIC = 4,
    QUEUE_TRANSPARENT = 5,
    QUEUE_MISS = 6,
    QUEUE_COUNT = 7
};

struct WavefrontState
{
    Ray ray;

    float3 throughput = make_float3(1.f);
    float3 radiance = make_float3(0.f);

    RNG rng;

    int pixelIndex;
    uint depth = 0;

    bool isInside = false;
    bool lastBounceWasDelta = false;
    float lastBsdfPdf = 1.f;

    HD WavefrontState(const Ray &r, const RNG &random, int pixel)
        : ray(r), throughput(make_float3(1.f)), radiance(make_float3(0.f)), rng(random), pixelIndex(pixel), depth(0), isInside(false), lastBounceWasDelta(false), lastBsdfPdf(1.f)
    {
    }

};