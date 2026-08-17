#pragma once

#include "../utils/object_type.h"
#include "../utils/op.cuh"

struct Payload
{
    int hit;

    float t;

    float3 position;
    float3 normal;
    float3 contribution = make_float3(0.f);
    int materialIndex;
    int depth = 0;
    
    float3 bsdfDir = make_float3(0.f);
    int isInside = 0;
    float bsdfPdf = 0.f;
    int lastBounceWasDelta = 0;  // 1 if last bounce was from mirror/transparent
};

struct ShadowPayload
{
    float3 transmittance;
    int depth;
};