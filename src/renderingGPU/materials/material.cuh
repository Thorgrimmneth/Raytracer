#pragma once

#include "../../defines.hpp"
#include "../raytracingUtils/ray.cuh"
#include "../raytracingUtils/hitrecord.cuh"
#include "../lights/lightsample.cuh"
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
    float4 baseColor; // xyz=color, w=type
    float4 params;    // x=roughness, y=metalness, z=ior, w=emission/intensity/shininess

    __host__
    static Material makeMaterial(
        float3 color,
        MaterialType type,
        float rough = 0.5f,
        float metal = 0.f,
        float ior = 1.5f,
        float emission = 0.f)
    {
        Material m;
        m.baseColor = make_float4(color.x, color.y, color.z, (float)type);
        m.params    = make_float4(rough, metal, ior, emission);
        return m;
    }

    __host__
    static Material randomMetal()
    {
        float3 color = make_float3(
            0.5f + 0.5f * RT::randomFloat(),
            0.5f + 0.5f * RT::randomFloat(),
            0.5f + 0.5f * RT::randomFloat()
        );

        float metal = 0.8f + 0.2f * RT::randomFloat();
        float rough = RT::randomFloat();

        return makeMaterial(color, METAL, rough, metal);
    }

    // ==== GETTERS ====
    __host__ __device__
    inline float3 color() const { return make_float3(baseColor); }

    __host__ __device__
    inline MaterialType type() const { return (MaterialType)((int)baseColor.w); }

    __host__ __device__
    inline float roughness() const { return params.x; }

    __host__ __device__
    inline float metalness() const { return params.y; }

    __host__ __device__
    inline float ior() const { return params.z; }

    __host__ __device__
    inline float emission() const { return params.w; }

    __host__ __device__
    inline float intensity() const { return params.w; }

    // ==== LAMBERT ====
    __device__ 
    inline float3 evaluateLambert() const;

    __device__
    float3 samplingLambert(const float3 normal, curandState* rngStates) const;

    __device__
    float pdfLambert(const float3 normal, const float3 direction) const;

    // ==== GGX ====
    __device__ 
    float computeD(const float3 &p_normal, const float3 &h) const;

    __device__
    float3 computeF(const float3 &wo, const float3 &h, const float3 &F0) const;

    __device__ 
    float computeG1(const float &x, const float &k) const;

    __device__ 
    float computeG(const float3 &wi, const float3 &wo, const float3 &p_normal) const;

    __device__ 
    inline float3 evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const;

    __device__ 
    float3 samplingGGX(const float3 &wo, const float3 &normal, curandState* rngStates) const;

    __device__ 
    float pdfGGX(const float3 normal, const float3 direction, const float3 wo) const;

    // ==== UTILS ====
    __device__ 
    float3 toWorld(const float3& normal, const float3 direction) const;

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