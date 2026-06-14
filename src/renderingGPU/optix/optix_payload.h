#pragma once

#include "../utils/objectType.h"

struct Payload
{
    int hit;

    float t;

    float4 position;
    float3 normal;

    int objectIndex;
    int materialIndex;

    HitObjectType objectType;
};