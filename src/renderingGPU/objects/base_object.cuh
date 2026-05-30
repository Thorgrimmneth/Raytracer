#pragma once

#include "../objectsUtils/aabb.cuh"
#include "../utils/macro.cuh"
#include <stdint.h>

enum ObjectType : uint32_t
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

    HD BaseObject(float3 min, float3 max, ObjectType type, int index)
        : minType(make_float4(min, intBitsToFloat(int(type)))), maxIndex(make_float4(max, intBitsToFloat(index)))
    {
    }

    HD float3 getMin() const { return make_float3(minType); }

    HD float3 getMax() const { return make_float3(maxIndex); }

    HD ObjectType getType() const { return ObjectType(floatBitsToInt(minType.w)); }

    HD int getIndex() const { return floatBitsToInt(maxIndex.w); }
};