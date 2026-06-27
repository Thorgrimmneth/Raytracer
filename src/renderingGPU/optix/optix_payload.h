#pragma once

#include "../utils/objectType.h"

struct Payload
{
    int hit;

    float t;

    float3 position;
    float3 normal;

    int objectIndex;
    int materialIndex;

    HitObjectType objectType;
};

struct ShadowPayload
{
    float3 transmittance;
};