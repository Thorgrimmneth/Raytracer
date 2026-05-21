#pragma once

#include <stdint.h>
#include "../objectsUtils/aabb.cuh"
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
    DataMin dataMin;
    DataMax dataMax;

    BaseObject(float3 min, float3 max, ObjectType type, int index) : dataMin(DataMin{min, type}), dataMax(DataMax{max, index})
    {
    }

    HD float3 getMin() const { return dataMin.min; };

    HD float3 getMax() const { return dataMax.max; };

    D_FORCEINLINE ObjectType getType() const { return dataMin.type; };

    D_FORCEINLINE int getIndex() const { return dataMax.index; };
};