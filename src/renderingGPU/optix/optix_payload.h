#pragma once

#include "../utils/object_type.h"

struct Payload
{
    int hit;

    float t;

    float3 position;
    float3 normal;

    int objectIndex;
    int materialIndex;

    Hitobject_type object_type;
};

struct ShadowPayload
{
    float3 transmittance;
    int depth;
};