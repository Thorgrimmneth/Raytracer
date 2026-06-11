#pragma once

#include "../../utils/defines.hpp"
#include "../utils/cuda_defines.cuh"
#include "../utils/macro.cuh"
#include "../utils/rng.cuh"

#include "../lights/lightsample.cuh"
#include "../raytracingUtils/ray.cuh"

#include "../../../devicePrograms/launch_params.cuh"

enum MaterialType
{
    LAMBERT,
    METAL,
    PLASTIC,
    TRANSPARENT,
    EMISSIVE,
    MIRROR
};

struct BSDFVal
{
    float3 brdf = make_float3(0.f);
    float3 direction = make_float3(0.f);
    float pdf = -1.f;

    bool isDelta;
};

struct Material
{
    float4 baseColor; // xyz=color, w=type
    float4 params;    // x=roughness, y=metalness, z=ior, w=emission/intensity/shininess

    HOST static Material makeMaterial(float3 color = make_float3(1.f), MaterialType type = LAMBERT, float rough = 0.5f,
                                      float metal = 0.f, float ior = 1.5f, float emission = 0.f)
    {
        Material m;
        m.baseColor = make_float4(color.x, color.y, color.z, (float)type);
        m.params = make_float4(rough, metal, ior, emission);
        return m;
    }

    HOST static Material randomMetal()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        float rough = 0.03f + 0.35f * randomFloat();

        return makeMaterial(color, METAL, rough, 1.f, 1.5f, 0.f);
    }

    HOST static Material randomLambert()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        return makeMaterial(color, LAMBERT, 1.0f, 0.f, 1.5f, 0.f);
    }

    HOST static Material randomPlastic()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        float rough = 0.05f + 0.25f * randomFloat();

        return makeMaterial(color, PLASTIC, rough, 0.f, 1.5f, 0.f);
    }

    HOST static Material randomTransparent()
    {
        float3 color =
            make_float3(0.25f + 0.75f * randomFloat(), 0.25f + 0.75f * randomFloat(), 0.25f + 0.75f * randomFloat());

        float ior = 1.25f + 0.35f * randomFloat();

        return makeMaterial(color, TRANSPARENT, 0.f, 0.f, ior, 0.f);
    }

    HOST static Material randomEmissive()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        float intensity = 2.f + 9.f * randomFloat();

        return makeMaterial(color, EMISSIVE, 0.f, 0.f, 1.f, intensity);
    }
    // ==== GETTERS ====
    HD_INLINE float3 color() const { return make_float3(baseColor); }

    HD_INLINE MaterialType type() const { return (MaterialType)((int)baseColor.w); }

    HD_INLINE float roughness() const { return params.x; }

    HD_INLINE float metalness() const { return params.y; }

    HD_INLINE float ior() const { return params.z; }

    HD_INLINE float emission() const { return params.w; }

    HD_INLINE float intensity() const { return params.w; }

    // ==== LAMBERT ====
    DEVICE float3 evaluateLambert() const;

    DEVICE float3 samplingLambert(const float3 normal, RNG *rngStates) const;
    DEVICE float pdfLambert(const float3 normal, const float3 direction) const;

    // ==== GGX ====
    DEVICE float computeD(const float3 &p_normal, const float3 &h) const;

    DEVICE float3 computeF(const float3 &wo, const float3 &h, const float3 &F0) const;

    DEVICE float computeG1(const float &NdotV) const;

    DEVICE float computeG(const float3 &wi, const float3 &wo, const float3 &n) const;

    DEVICE float3 evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const;

    DEVICE float3 samplingGGX(const float3 &wo, const float3 &normal, RNG *rngStates) const;

    DEVICE float pdfGGX(const float3 n, const float3 direction, const float3 wo) const;

    // ==== UTILS ====
    DEVICE float3 toWorld(const float3 &normal, const float3 direction) const;

    DEVICE void createONB(const float3 &n, float3 &tangent, float3 &bitangent) const;

    D_FORCEINLINE
    float3 fresnelSchlick(float cosTheta, const float3& F0) const
    {
        float m  = saturate(1.f - fabsf(cosTheta));
        float m2 = m * m;
        float m5 = m2 * m2 * m;
        return F0 + (make_float3(1.f) - F0) * m5;
    }

    D_FORCEINLINE
    float3 computeTransmission() const
    {
        // Energy-based transmission: Fresnel at normal incidence
        // Uses IOR to compute the reflection coefficient at normal angle
        float ior = this->ior();
        float eta = 1.f / ior;  // ratio of refraction indices (air to material)
        float r0 = (1.f - eta) / (1.f + eta);  // reflection coefficient at normal incidence
        r0 *= r0;
        float transmission = 1.f - r0;  // transmission = 1 - reflection
        
        // Apply material color to the transmission
        return color() * transmission;
    }

    DEVICE BSDFVal getLambertBSDF(const Ray &ray, const OptixHit &hit, RNG *rngStates) const;

    DEVICE BSDFVal getMetalBSDF(const Ray &ray, const OptixHit &hit, RNG *rngStates) const;

    DEVICE BSDFVal getPlasticBSDF(const Ray &ray, const OptixHit &hit, RNG *rngStates) const;

    DEVICE BSDFVal getMirrorBSDF(const Ray &ray, const OptixHit &hit) const;

    DEVICE BSDFVal getTransparentBSDF(const Ray &ray, const OptixHit &hit, RNG *rngStates, bool &isInside) const;

    DEVICE BSDFVal getBSDF(const Ray &ray, const OptixHit &hit, RNG *rngStates, bool &isInside) const;

    DEVICE float3 evalLambertBSDF() const;

    DEVICE float3 evalMetalBSDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float3 evalPlasticBSDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float3 evalBSDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float lambertPDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float metalPDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float plasticPDF(const Ray &ray, const OptixHit &hit, const float3 &wi) const;

    DEVICE float pdf(const Ray &ray, const OptixHit &hit, const float3 &wi) const;
};