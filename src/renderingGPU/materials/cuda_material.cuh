#pragma once

#include "../../defines.hpp"
#include "../raytracingUtils/cuda_ray.cuh"
#include "../raytracingUtils/cuda_hitrecord.cuh"
#include "../lights/cuda_lightsample.cuh"
#include "../utils/cuda_defines.cuh"
#include <curand_kernel.h>

enum MaterialType
{
    LAMBERT,
    EMISSIVE,
    METAL,
    MIRROR,
    PLASTIC,
    TRANSPARENT
};

struct BSDFVal
{
    float3 brdf = make_float3(0.f);
    float3 direction = make_float3(0.f);
    float pdf = -1.f;
};

struct Material
{
    float3 color;
    float intensity = 1.f;
    float metalness = 0.f;
    float alpha = 0.f;        // GGX alpha = roughness^2
    float ruggedness = 0.f;   // roughness in [0,1]
    float ior = 1.f;
    float shininess = 1.f;
    MaterialType type;

    Material() = default;
    Material(float3 c) : color(c), type(MaterialType::LAMBERT){}
    Material(float3 c, bool isMirror) : color(c), type(MaterialType::MIRROR){}
    Material(float3 c, float i, MaterialType t) : color(c), intensity(i){
        switch (t)
        {
            case MaterialType::PLASTIC:
                shininess = i;
                break;
            case MaterialType::TRANSPARENT:
                ior = i;
                break;
            case MaterialType::EMISSIVE:
                intensity = i;
                break;
        }
        type = t;
    }
    Material(float3 c, float m, float r) : color(c), metalness(m), ruggedness(r), alpha(max(r, 0.f) * max(r, 0.f)), type(MaterialType::METAL){}

    __host__
    static inline Material randomMetal()
    {
        float3 color = make_float3(
            0.5f + 0.5f * RT::randomFloat(),
            0.5f + 0.5f * RT::randomFloat(),
            0.5f + 0.5f * RT::randomFloat()
        );

        float metalness = 0.8f + 0.2f * RT::randomFloat();
        float ruggedness = RT::randomFloat();

        return Material(color, metalness, ruggedness);
    }
    __device__ 
    inline float3 evaluateLambert() const;

    __device__
    float3 samplingLambert(const float3 normal, curandState* rngStates) const;

    __device__
    float pdfLambert(const float3 normal, const float3 direction) const;

    __device__ 
    float computeD(const float3 &p_normal, const float3 &h) const;

    __device__
    float3 computeF(const float3 &wo, const float3 &h, const float3 &F0) const;

    __device__ float computeG1(const float &x, const float &k) const;

    __device__ float computeG(const float3 &wi, const float3 &wo, const float3 &p_normal) const;

    __device__ inline float3 evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const;
    __device__ float3 samplingGGX(const float3 &wo, const float3 &normal, curandState* rngStates) const;
    __device__ float pdfGGX(const float3 normal, const float3 direction, const float3 wo) const;

    __device__ float3 toWorld(const float3& normal, const float3 direction) const;
    __device__ void createONB(const float3& n, float3& tangent, float3& bitangent) const;

    __device__ float3 fresnelSchlick(float cosTheta, const float3& F0) const;
    
    __device__
    BSDFVal getBSDF(
        const Ray &ray,
        const HitRecord &hit,
        curandState* rngStates,
        bool &isInside,
        bool &reflected) const;

    __device__
    BSDFVal getBSDF(
        const Ray &ray,
        const HitRecord &hit,
    curandState* rngStates) const;

    __device__
    float3 evalBSDF(
        const Ray &ray,
        const HitRecord &hit,
        const float3 &wi
    ) const;

    __device__
    float pdf(
        const Ray &ray,
        const HitRecord &hit,
        const float3 &wi
    ) const;
};