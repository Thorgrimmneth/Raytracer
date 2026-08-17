#pragma once

#include "../utils/object_type.h"
#include "../utils/op.cuh"

struct Payload
{
    int hit;

    float t;

    float3 position;
    float3 normal;
    float3 accumulated_color = make_float3(0.f);
    float3 luminous_contribution = make_float3(0.f);
    int objectIndex;
    int materialIndex;
    float hitDistance;

    Hitobject_type object_type;
    
    float3 bsdfDir = make_float3(0.f);
    float bsdfPdf = 0.f;
};

struct ShadowPayload
{
    float3 transmittance;
    int depth;
};