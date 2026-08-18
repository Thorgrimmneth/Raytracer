#pragma once

#include "../../utils/rng_cpu.hpp"
#include "../utils/defines.cuh"
#include "../utils/op.cuh"
#include "../utils/rng.cuh"
#include "../utils/simplified_def.cuh"

enum MaterialType : int
{
    MISS = 0,
    LAMBERT,
    METAL,
    PLASTIC,
    TRANSPARENT,
    EMISSIVE,
    MIRROR,

    MATERIAL_TYPE_COUNT
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
    float4 params;    // x=alpha, y=metalness, z=ior, w=emission/intensity/shininess

    HD_INLINE Material() : baseColor(make_float4(1.f, 1.f, 1.f, (float)MISS)), params(make_float4(1.f, 0.f, 1.5f, 0.f))
    {
    }
    HOST static Material makeMaterial(float3 color = make_float3(1.f), MaterialType type = LAMBERT, float alpha = 0.5f,
                                      float metal = 0.f, float ior = 1.5f, float emission = 0.f)
    {
        Material m;
        m.baseColor = make_float4(color.x, color.y, color.z, (float)type);
        m.params = make_float4(alpha, metal, ior, emission);
        return m;
    }

    HOST static Material randomMetal()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        float rough = 0.03f + 0.35f * randomFloat();

        return makeMaterial(color, METAL, rough * rough, 1.f);
    }

    HOST static Material randomLambert()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        return makeMaterial(color, LAMBERT, 1.0f, 0.f);
    }

    HOST static Material randomPlastic()
    {
        float3 color = make_float3(randomFloat(), randomFloat(), randomFloat());

        float rough = 0.05f + 0.25f * randomFloat();

        return makeMaterial(color, PLASTIC, rough * rough, 0.f, 1.5f, 0.f);
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

    HD_INLINE float alpha() const { return params.x; }

    HD_INLINE float metalness() const { return params.y; }

    HD_INLINE float ior() const { return params.z; }

    HD_INLINE float emission() const { return params.w; }

    HD_INLINE float intensity() const { return params.w; }

    DEVICE void createONB(const float3 &n, float3 &tangent, float3 &bitangent) const
    {
        if (n.z < -0.999f)
        {
            tangent = make_float3(0.0f, -1.0f, 0.0f);
            bitangent = make_float3(-1.0f, 0.0f, 0.0f);
            return;
        }

        float a = 1.0f / (1.0f + n.z);
        float b = -n.x * n.y * a;

        tangent = make_float3(1.0f - n.x * n.x * a, b, -n.x);
        bitangent = make_float3(b, 1.0f - n.y * n.y * a, -n.y);
    }

    DEVICE float3 toWorld(const float3 &normal, const float3 direction) const
    {
        float3 T, B;
        createONB(normal, T, B);
        return normalize(direction.x * T + direction.y * B + direction.z * normal);
    }

    // ------------------------------------------------------------
    //  Lambert
    // ------------------------------------------------------------

    DEVICE inline float3 evaluateLambert() const { return color() * GPUInvPIf; }

    DEVICE float3 samplingLambert(const float3 normal, RNG &rngStates) const
    {
        float e1 = rngStates.nextFloat();
        float e2 = rngStates.nextFloat();
        float r = sqrtf(e1);
        float phi = 2.f * GPUPIf * e2;
        float x = r * cosf(phi);
        float y = r * sinf(phi);
        float z = sqrtf(fmaxf(0.f, 1.f - x * x - y * y));
        return toWorld(normal, make_float3(x, y, z));
    }

    DEVICE float pdfLambert(const float3 normal, const float3 direction) const
    {
        float cosTheta = fmaxf(dot(normal, direction), 0.f);
        return cosTheta * GPUInvPIf;
    }

    // ------------------------------------------------------------
    //  GGX / Cook-Torrance
    // ------------------------------------------------------------

    DEVICE float computeD(const float3 &p_normal, const float3 &h, const float alphaSquared) const
    {
        float NdotH = fmaxf(dot(p_normal, h), 0.f);
        float NdotH2 = NdotH * NdotH;
        float denom = (NdotH2 * (alphaSquared - 1.f) + 1.f);
        denom = 1.f / (GPUPIf * denom * denom);
        return alphaSquared * denom;
    }

    DEVICE float3 computeF(const float3 &wo, const float3 &h, const float3 &F0) const
    {
        float HdotV = clamp(dot(h, wo), 0.f, 1.f);
        float m = 1.f - HdotV;
        float m2 = m * m;
        float m5 = m2 * m2 * m;
        return F0 + (make_float3(1.f) - F0) * m5;
    }

    DEVICE float computeG1(const float &NdotV, const float alphaSquared) const
    {
        if (NdotV <= 0.f)
            return 0.f;
        float NdotV2 = NdotV * NdotV;
        float tan2 = (1.f - NdotV2) / fmaxf(NdotV2, 1e-8f);

        return 2.f / (1.f + sqrtf(1.f + alphaSquared * tan2));
    }

    DEVICE float computeG(const float &NdotV, const float &NdotL, const float alphaSquared) const
    {

        return computeG1(NdotV, alphaSquared) * computeG1(NdotL, alphaSquared);
    }

    DEVICE float3 evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const
    {
        float3 h = normalize(wi + wo);
        float NdotV = fmaxf(dot(normal, wo), 0.f);
        float NdotL = fmaxf(dot(normal, wi), 0.f);

        if (NdotV <= 0.f || NdotL <= 0.f)
            return make_float3(0.f);

        float alphaT = alpha();
        float alphaSquared = alphaT * alphaT;
        float D = computeD(normal, h, alphaSquared);
        float3 F = computeF(wo, h, F0);
        float G = computeG(NdotV, NdotL, alphaSquared);

        float denom = 4.f * NdotV * NdotL;
        if (denom < 1e-6f)
            return make_float3(0.f);

        return (D * G / denom) * F;
    }

    DEVICE float3 samplingGGX(const float3 &wo, const float3 &normal, RNG &rngStates) const
    {
        float a = fmaxf(alpha(), 1e-4f);

        float3 T, B;
        createONB(normal, T, B);

        float3 V = normalize(make_float3(dot(wo, T), dot(wo, B), dot(wo, normal)));

        V = normalize(make_float3(a * V.x, a * V.y, V.z));

        float3 T1 = (V.z < 0.9999f) ? normalize(cross(make_float3(0.f, 0.f, 1.f), V)) : make_float3(1.f, 0.f, 0.f);
        float3 T2 = cross(V, T1);

        float e1 = rngStates.nextFloat();
        float e2 = rngStates.nextFloat();
        float r = sqrtf(e1);
        float phi = 2.f * GPUPIf * e2;

        float t1 = r * cosf(phi);
        float t2 = r * sinf(phi);

        float s = 0.5f * (1.f + V.z);
        t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

        float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * V;

        float3 h = normalize(make_float3(a * Nh.x, a * Nh.y, fmaxf(0.f, Nh.z)));
        h = toWorld(normal, h);

        float3 wi = reflect(-wo, h);

        if (dot(normal, wi) <= 0.f)
            return make_float3(0.f);

        return wi;
    }

    DEVICE float pdfGGX(const float3 n, const float3 wi, const float3 wo) const
    {
        float3 h = normalize(wi + wo);

        if (dot(wo, h) <= 0.f)
            return 0.f;

        float NdotV = fmaxf(dot(n, wo), 0.f);

        if (NdotV <= 0.f)
            return 0.f;
        float alphaT = alpha();
        float alphaSquared = alphaT * alphaT;
        float D = computeD(n, h, alphaSquared);
        float G1 = computeG1(NdotV, alphaSquared);
        float NdotH = fmaxf(dot(n, h), 0.f);
        float VdotH = fmaxf(dot(wo, h), 0.f);

        if (NdotH <= 0.f || VdotH <= 0.f)
            return 0.f;

        return D * G1 * NdotH / (4.f * NdotV * VdotH);
    }

    // ============================================================
    //  getBSDF helpers
    // ============================================================

    DEVICE BSDFVal getMetalBSDF(const float3 &direction, const float3 &normal, RNG &rngStates) const
    {
        if (alpha() < 1e-4f)
        {
            BSDFVal bsdf;
            bsdf.direction = reflect(direction, normal);
            bsdf.pdf = 1.f;
            bsdf.brdf = color();
            return bsdf;
        }

        float3 wo = -direction;
        BSDFVal bsdf;

        float3 F0 = lerp(make_float3(0.04f), color(), metalness());

        bsdf.direction = samplingGGX(wo, normal, rngStates);

        if (dot(normal, bsdf.direction) <= 0.f)
        {
            bsdf.pdf = 0.f;
            bsdf.brdf = make_float3(0.f);
            return bsdf;
        }

        bsdf.pdf = pdfGGX(normal, bsdf.direction, wo);
        bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);

        return bsdf;
    }

    DEVICE BSDFVal getLambertBSDF(const float3 &direction, const float3 &normal, RNG &rngStates) const
    {
        BSDFVal bsdf;

        bsdf.direction = samplingLambert(normal, rngStates);
        bsdf.pdf = pdfLambert(normal, bsdf.direction);
        bsdf.brdf = evaluateLambert();

        return bsdf;
    }

    DEVICE BSDFVal getPlasticBSDF(const float3 &direction, const float3 &normal, RNG &rngStates) const
    {
        float3 wo = -direction;
        BSDFVal bsdf;
        float cosTheta = saturate(dot(normal, wo));
        float3 F0 = make_float3(0.04f);
        float3 F = fresnelSchlick(cosTheta, F0);

        float specW = (F.x + F.y + F.z) / 3.f;

        if (rngStates.nextFloat() < specW)
        {
            bsdf.direction = samplingGGX(wo, normal, rngStates);

            if (dot(normal, bsdf.direction) <= 0.f)
            {
                bsdf.pdf = 0.f;
                bsdf.brdf = make_float3(0.f);
                return bsdf;
            }

            bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);
        }
        else
        {
            bsdf.direction = samplingLambert(normal, rngStates);
            bsdf.brdf = evaluateLambert();
        }
        bsdf.pdf = specW * pdfGGX(normal, bsdf.direction, wo) + (1.f - specW) * pdfLambert(normal, bsdf.direction);

        return bsdf;
    }

    DEVICE BSDFVal getMirrorBSDF(const float3 &direction, const float3 &normal) const
    {
        BSDFVal bsdf;

        bsdf.direction = reflect(direction, normal);
        bsdf.pdf = 1.f;
        bsdf.brdf = color();

        return bsdf;
    }

    DEVICE BSDFVal getTransparentBSDF(const float3 &direction, const float3 &normal, RNG &rngStates,
                                      int &isInside) const
    {
        float3 wo = -direction;
        BSDFVal bsdf;

        float3 n = normal;
        float cosI = dot(n, wo);

        if (cosI < 0.f)
        {
            n = -n;
            cosI = -cosI;
        }

        float ior = this->ior();

        float n1 = isInside ? ior : 1.f;
        float n2 = isInside ? 1.f : ior;
        float eta = n1 / n2;

        float k = 1.f - eta * eta * (1.f - cosI * cosI);

        // Total internal reflection
        if (k < 0.f)
        {
            bsdf.direction = reflect(-wo, n);
            bsdf.pdf = 1.f;
            bsdf.brdf = make_float3(1.f);
            return bsdf;
        }

        float cosT = sqrtf(k);

        float rs = ((n1 * cosI) - (n2 * cosT)) / ((n1 * cosI) + (n2 * cosT));
        rs *= rs;

        float rp = ((n2 * cosI) - (n1 * cosT)) / ((n2 * cosI) + (n1 * cosT));
        rp *= rp;

        float reff = 0.5f * (rs + rp);
        float xi = rngStates.nextFloat();

        if (xi < reff)
        {
            bsdf.direction = reflect(-wo, n);
            bsdf.pdf = reff;
            bsdf.brdf = make_float3(1.f);
        }
        else
        {
            float3 wi = -eta * (wo) + (eta * cosI - cosT) * n;

            bsdf.direction = normalize(wi);
            bsdf.pdf = 1.f - reff;
            bsdf.brdf = color();

            isInside = !isInside;
        }

        return bsdf;
    }

    // ============================================================
    //  getBSDF dispatcher
    // ============================================================

    DEVICE BSDFVal getBSDF(const float3 &direction, const float3 &normal, RNG &rngStates, int &isInside) const
    {
        switch (type())
        {
        case LAMBERT:
            return getLambertBSDF(direction, normal, rngStates);

        case METAL:
            return getMetalBSDF(direction, normal, rngStates);

        case PLASTIC:
            return getPlasticBSDF(direction, normal, rngStates);

        case MIRROR:
            return getMirrorBSDF(direction, normal);

        case TRANSPARENT:
            return getTransparentBSDF(direction, normal, rngStates, isInside);

        default: {
            BSDFVal bsdf;
            bsdf.direction = make_float3(0.f);
            bsdf.pdf = 0.f;
            bsdf.brdf = make_float3(0.f);
            return bsdf;
        }
        }
    }

    // ============================================================
    //  evalBSDF helpers
    //  Used by NEE / MIS — delta materials return 0
    // ============================================================

    DEVICE float3 evalLambertBSDF() const { return evaluateLambert(); }

    DEVICE float3 evalMetalBSDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        float3 wo = -direction;

        if (dot(normal, wi) <= 0.f)
            return make_float3(0.f);

        float3 F0 = lerp(make_float3(0.04f), color(), metalness());

        return evaluateGGX(wo, normal, wi, F0);
    }

    DEVICE float3 evalPlasticBSDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        float3 wo = -direction;

        if (dot(normal, wi) <= 0.f)
            return make_float3(0.f);

        float3 F0 = make_float3(0.04f);
        float cosTheta = saturate(dot(normal, wo));
        float3 F = fresnelSchlick(cosTheta, F0);

        float3 diffuse = evaluateLambert();
        float3 specular = evaluateGGX(wo, normal, wi, F0);

        return (make_float3(1.f) - F) * diffuse + specular;
    }

    // ============================================================
    //  evalBSDF dispatcher
    // ============================================================

    DEVICE float3 evalBSDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        switch (type())
        {
        case LAMBERT:
            return evalLambertBSDF();

        case METAL:
            return evalMetalBSDF(direction, normal, wi);

        case PLASTIC:
            return evalPlasticBSDF(direction, normal, wi);

        // Delta materials have no continuous BSDF for NEE / MIS
        case MIRROR:
        case TRANSPARENT:
        default:
            return make_float3(0.f);
        }
    }

    // ============================================================
    //  pdf helpers
    //  Used by MIS — delta materials return 0
    // ============================================================

    DEVICE float lambertPDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {

        return pdfLambert(normal, wi);
    }

    DEVICE float metalPDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        float3 wo = -direction;

        if (dot(normal, wi) <= 0.f)
            return 0.f;

        return pdfGGX(normal, wi, wo);
    }

    DEVICE float plasticPDF(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        float3 wo = -direction;

        if (dot(normal, wi) <= 0.f)
            return 0.f;

        float3 F0 = make_float3(0.04f);
        float cosTheta = saturate(dot(normal, wo));
        float3 F = fresnelSchlick(cosTheta, F0);
        float specW = (F.x + F.y + F.z) / 3.f;

        return specW * pdfGGX(normal, wi, wo) + (1.f - specW) * pdfLambert(normal, wi);
    }

    // ============================================================
    //  pdf dispatcher
    // ============================================================

    DEVICE float pdf(const float3 &direction, const float3 &normal, const float3 &wi) const
    {
        switch (type())
        {
        case LAMBERT:
            return lambertPDF(direction, normal, wi);

        case METAL:
            return metalPDF(direction, normal, wi);

        case PLASTIC:
            return plasticPDF(direction, normal, wi);

        // Delta materials are sampled discretely, so no continuous PDF here
        case MIRROR:
        case TRANSPARENT:
        default:
            return 0.f;
        }
    }

    DEVICE float3 fresnelSchlick(float cosTheta, const float3 &F0) const
    {
        float m = saturate(1.f - fabsf(cosTheta));
        float m2 = m * m;
        float m5 = m2 * m2 * m;
        return F0 + (make_float3(1.f) - F0) * m5;
    }

    DEVICE float3 computeTransmission() const
    {
        // Energy-based transmission: Fresnel at normal incidence
        // Uses IOR to compute the reflection coefficient at normal angle
        float eta = 1.f / this->ior();        // ratio of refraction indices (air to material)
        float r0 = (1.f - eta) / (1.f + eta); // reflection coefficient at normal incidence
        r0 *= r0;
        float transmission = 1.f - r0; // transmission = 1 - reflection

        // Apply material color to the transmission
        return color() * transmission;
    }
};