#pragma once

#include "../objectsUtils/aabb.cuh"
#include <stdint.h>
#include "../utils/macro.cuh"

enum ObjectType : uint32_t
{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT_SPHERE
};

struct DataMin
{
    float3 min;
    ObjectType type;
};

struct DataMax
{
    float3 max;
    int index;
};

struct BaseObject
{
    DataMin min;
    DataMax max;

    BaseObject(float3 min, float3 max, ObjectType type, int index) : min(DataMin{min, type}), max(DataMax{max, index})
    {
    }

    HD float3 getMin() const { return min.min; };

    HD float3 getMax() const { return max.max; };

    D_FORCEINLINE ObjectType getType() const { return min.type; };

    D_FORCEINLINE int getIndex() const { return max.index; };
};