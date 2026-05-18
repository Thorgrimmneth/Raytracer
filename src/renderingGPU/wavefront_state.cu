#pragma once

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

    float3 throughput;
    float3 radiance;

    RNG rng;

    int pixelIndex;
    int depth;
    bool active;

    bool isInside;
    int lastBounceWasDelta;
    float lastBsdfPdf;

    __host__ __device__
    WavefrontState(
        const Ray& r,
        const RNG& random,
        int pixel)
        : ray(r),
          throughput(make_float3(1.f)),
          radiance(make_float3(0.f)),
          rng(random),
          pixelIndex(pixel),
          depth(0),
          active(1),
          isInside(false),
          lastBounceWasDelta(1),
          lastBsdfPdf(1.f)
    {}

    __host__ __device__
    void terminate()
    {
        active = false;
    }
};