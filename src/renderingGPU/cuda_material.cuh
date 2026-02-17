#pragma once

#include "objectGPU/cuda_ray.cuh"
#include "objectGPU/cuda_hitrecord.cuh"
#include "objectGPU/cuda_lightsample.cuh"
#include "cuda_defines.cuh"

enum MaterialType
{
    COLOR,
    LAMBERT,
    EMISSIVE,
    METAL,
    MIRROR,
    PLASTIC,
    TRANSPARENT
};

struct Material
{
    float3 color;
    float intensity = 1.f;
    float metalness = 0.f;
    float alpha = 0.f;
    float ruggedness = 0.f;
    float ior = 1.f;
    float shininess = 1.f;
    MaterialType type;

    __device__ 
    inline float3 evaluateLambertBRDF();

    __device__ 
    float computeD(const float3 &p_normal, const float3 &h) const;

    __device__
    float3 computeF(const float3 &wo, const float3 &h, const float3 &F0) const;

    __device__ float computeG1(const float &x, const float &k) const;

    __device__ float computeG(const float3 &wi, const float3 &wo, const float3 &p_normal) const;

    __device__ inline float3 evaluateCookTorranceBRDF(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const;

    __device__
    float3 getColor(
        const Ray &ray,
        const HitRecord &hit,
        const LightSample &light);
};