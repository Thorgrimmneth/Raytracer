#pragma once

#include "../utils/op.cuh"
#include "../utils/macro.cuh"

enum HitObjectType
{
    HIT_SPHERE,
    HIT_TRIANGLE_MESH,
    HIT_PLANE,
    HIT_SPHERE_IMPLICIT
};

struct HitRecord
{
    float3 point;
    float3 normal;
    float  distance;

    int materialIndex;

    int objectIndex;
    HitObjectType objectType;

    D_FORCEINLINE
    void faceNormal(const float3 &direction)
    {
        normal = dot(direction, normal) < 0.f ? normal : -normal; 
    }
};