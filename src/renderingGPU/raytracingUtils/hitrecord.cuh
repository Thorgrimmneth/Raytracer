#pragma once

#include "../utils/macro.cuh"
#include "../utils/op.cuh"
#include "../utils/objectType.h"

struct HitRecord
{
    float4 point;  // xyz = point, w = distance
    float4 normal; // xyz = normal, w = materialIndex

    int objectIndex;
    HitObjectType objectType;

    D_FORCEINLINE float getDistance() const { return point.w; }

    D_FORCEINLINE float3 getPoint() const { return make_float3(point); }

    D_FORCEINLINE float3 getNormal() const { return make_float3(normal); }

    D_FORCEINLINE int getMaterialIndex() const { return floatBitsToInt(normal.w); }

    D_FORCEINLINE void setMaterialIndex(int materialIndex) { normal.w = intBitsToFloat(materialIndex); }
    
    D_FORCEINLINE void setHitInfo(const float3 &p, const float3 &n, float distance, int materialIndex, int objIndex,
                                  HitObjectType objType)
    {
        point = make_float4(p, distance);
        normal = make_float4(n, intBitsToFloat(materialIndex));
        objectIndex = objIndex;
        objectType = objType;
    }

    D_FORCEINLINE void faceNormal(const float3 &direction) { normal = dot(direction, make_float3(normal)) < 0.f ? normal : -normal; }
};