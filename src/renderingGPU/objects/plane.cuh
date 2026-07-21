#pragma once

#include "../materials/material.cuh"
#include "../../devicePrograms/launch_radiance_params.cuh"
struct Plane
{
    float4 normal; // xyz = normal, w = delta
    int materialIndex;

    Plane() = default;
    Plane(float3 pos, float3 n) : normal(make_float4(normalize(n), dot(normalize(-n), pos))), materialIndex(0) {}

    D_FORCEINLINE
    float3 getNormal() const { return make_float3(normal); }

    D_FORCEINLINE
    float getDelta() const { return normal.w; }
};