#pragma once

#include "../objectsUtils/aabb.cuh"
#include "../utils/simplified_def.cuh"
#include <stdint.h>

enum object_type : uint32_t
{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT_SPHERE
};

struct BaseObject
{
    float4 minType;
    float4 maxIndex;

    HD BaseObject(float3 min, float3 max, object_type type, int index)
        : minType(make_float4(min, intBitsToFloat(int(type)))), maxIndex(make_float4(max, intBitsToFloat(index)))
    {
    }

    HD float3 getMin() const { return make_float3(minType); }

    HD float3 getMax() const { return make_float3(maxIndex); }

    HD object_type getType() const { return object_type(floatBitsToInt(minType.w)); }

    HD int getIndex() const { return floatBitsToInt(maxIndex.w); }
};