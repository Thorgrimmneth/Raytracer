#pragma once

#include "cuda_op.cuh"

struct HitRecord
{
    float3 point;
    float3 normal;
    float  distance;

    int materialIndex;

    __device__
    void faceNormal(const float3& direction);
};