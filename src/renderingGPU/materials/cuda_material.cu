#include "cuda_material.cuh"

__device__ 
inline float3 Material::evaluateLambertBRDF()
{
    return color * GPUInvPIf;
}

__device__ 
float Material::computeD(const float3 &p_normal, const float3 &h) const
{
    float alphaSquared = alpha * alpha;
    float NdotH = max(dot(p_normal, h), 0.f);
    return alphaSquared / (GPUPIf * ((NdotH * NdotH) * (alphaSquared - 1.f) + 1.f) * ((NdotH * NdotH) * (alphaSquared - 1.f) + 1.f));
}

__device__
float3 Material::computeF(const float3 &wo, const float3 &h, const float3 &F0) const
{
    float HdotV = clamp(dot(h, wo), 0.0f, 1.0f);
    return F0 + (float3f(1.f) - F0) * pow(1.f - HdotV, 5.f);
}

__device__ float Material::computeG1(const float &x, const float &k) const 
{ 
    return x / (x * (1.f - k) + k); 
}

__device__ float Material::computeG(const float3 &wi, const float3 &wo, const float3 &p_normal) const
{
    float k = ((ruggedness + 1.f) * (ruggedness + 1.f)) / 8.f;
    return computeG1(max(dot(p_normal, wo), 0.f), k) * computeG1(max(dot(p_normal, wi), 0.f), k);
}

__device__ 
inline float3 Material::evaluateCookTorranceBRDF(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const
{
    float3 h = normalize(wi + wo);

    float NdotV = max(dot(normal, wo), 0.f);
    float NdotL = max(dot(normal, wi), 0.f);

    if (NdotV <= 0.f || NdotL <= 0.f)
        return float3f(0.f);

    float D = computeD(normal, h);
    float3 F = computeF(wo, h, F0);
    float G = computeG(wi, wo, normal);

    float denominator = 4.f * NdotV * NdotL;

    if (denominator < 1e-6f)
        return float3f(0.f);

    return (D * G / denominator) * F;
}
__device__
float3 Material::getColor(
        const Ray &ray,
        const HitRecord &hit,
        const LightSample &light) const
{
    float3 normal = normalize(hit.normal);
    float3 wo = normalize(-ray.direction);
    float3 wi = normalize(light.direction);

    switch (type)
    {
    case COLOR:
        return color;

    case LAMBERT:
        return color * GPUInvPIf;

    case EMISSIVE:
        return color * intensity;

    case METAL:
    {
        float3 F0 = lerp(float3f(0.04f), color, metalness);

        float3 h = normalize(wi + wo);
        float3 F = computeF(wo, h, F0);

        float3 spec = evaluateCookTorranceBRDF(wo, normal, wi, F0);

        float3 kd = (float3f(1.f) - F) * (1.f - metalness);
        float3 diffuse = kd * color * GPUInvPIf;

        return diffuse + spec;
    }
    case PLASTIC:
    {
        float3 n = normal;
        float3 h = normalize(wi + wo);

        float NdotL = max(dot(n, wi), 0.f);
        float NdotV = max(dot(n, wo), 0.f);

        if (NdotL <= 0.f || NdotV <= 0.f)
            return float3f(0.0f);

        // -------- Fresnel approx diélectrique --------
        float3 F0 = float3f(0.04f);

        float HdotV = max(dot(h, wo), 0.f);
        float x = 1.f - HdotV;
        float x2 = x * x;
        float3 F = F0 + (float3f(1.f) - F0) * x2 * x2 * x;

        // -------- Blinn-Phong specular --------
        float NdotH = max(dot(n, h), 0.f);

        float specNorm = (shininess + 2.f) * (1.f / (2.f * GPUPIf));

        float3 spec = F * specNorm * powf(NdotH, shininess);

        // -------- Diffuse --------
        float3 kd = float3f(1.f) - F;
        float3 diffuse = kd * color * GPUInvPIf;

        return diffuse + spec;
    }
    default:
        return float3f(0.0f);
    };
};
