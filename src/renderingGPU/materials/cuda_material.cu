#include "cuda_material.cuh"

__device__ 
float3 Material::toWorld(const float3& normal, const float3 direction) const{
    float3 T, B;
    createONB(normal, T, B);
    return direction.x * T + direction.y * B + direction.z * normal;
}

__device__ 
inline float3 Material::evaluateLambert() const
{
    return color * GPUInvPIf;
}

__device__
float3 Material::samplingLambert(const float3 normal, curandState* rngStates) const
{
    float e1 = curand_uniform(rngStates);
    float e2 = curand_uniform(rngStates);
    float r = sqrt(e1);
    float phi = 2 * GPUPIf * e2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float3 direction = toWorld(normal, make_float3(x, y, sqrt(1 - x *x - y* y)));
}

__device__
float Material::pdfLambert(const float3 normal, const float3 direction) const
{
    return dot(normal, direction) * GPUInvPIf;
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
    return F0 + (make_float3(1.f) - F0) * pow(1.f - HdotV, 5.f);
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
inline float3 Material::evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi, const float3 &F0) const
{
    float3 h = normalize(wi + wo);

    float NdotV = max(dot(normal, wo), 0.f);
    float NdotL = max(dot(normal, wi), 0.f);

    if (NdotV <= 0.f || NdotL <= 0.f)
        return make_float3(0.f);

    float D = computeD(normal, h);
    float3 F = computeF(wo, h, F0);
    float G = computeG(wi, wo, normal);

    float denominator = 4.f * NdotV * NdotL;

    if (denominator < 1e-6f)
        return make_float3(0.f);

    return (D * G / denominator) * F;
}

__device__
void Material::createONB(const float3& n, float3& tangent, float3& bitangent) const
{
    if (n.z < -0.9999999f) {
        tangent = make_float3(0.0f, -1.0f, 0.0f);
        bitangent = make_float3(-1.0f, 0.0f, 0.0f);
        return;
    }

    float a = 1.0f / (1.0f + n.z);
    float b = -n.x * n.y * a;

    tangent = make_float3(
        1.0f - n.x * n.x * a,
        b,
        -n.x
    );

    bitangent = make_float3(
        b,
        1.0f - n.y * n.y * a,
        -n.y
    );
}

__device__
float3 Material::samplingGGX(const float3 &wo, const float3 &normal, curandState* rngStates) const
{
    float3 T, B;
    createONB(normal, T, B);

    float3 V = normalize(make_float3(
        dot(wo, T),
        dot(wo, B),
        dot(wo, normal)
    ));

    V = normalize(make_float3(ruggedness * V.x, ruggedness * V.y, V.z));

    float3 T1 = (V.z < 0.9999f) ? normalize(cross(make_float3(0,0,1), V)) : make_float3(1,0,0);
    float3 T2 = cross(V, T1);

    float e1 = curand_uniform(rngStates);
    float e2 = curand_uniform(rngStates);

    float r = sqrt(e1);
    float phi = 2.f * GPUPIf * e2;

    float t1 = r * cos(phi);
    float t2 = r * sin(phi);

    float s = 0.5f * (1.f + V.z);
    t2 = (1.f - s) * sqrt(1.f - t1*t1) + s * t2;

    float3 Nh = t1*T1 + t2*T2 + sqrt(max(0.f, 1.f - t1*t1 - t2*t2))*V;

    float3 h = normalize(make_float3(
        ruggedness * Nh.x,
        ruggedness * Nh.y,
        max(0.f, Nh.z)
    ));

    h = toWorld(normal, h);

    float3 wi = reflect(-wo, h);

    if(dot(normal, wi) <= 0.f) return make_float3(0);

    return wi;
}

__device__
float Material::pdfGGX(const float3 normal, const float3 wi, const float3 wo) const
{
    float3 h = normalize(wo + wi);
    float D = computeD(normal, h);
    return (D * dot(normal, h)) / (4.f * dot(wo, h));
}

__device__
float3 Material::fresnelSchlick(float cosTheta, const float3& F0) const
{
    float m = saturate(1.f - cosTheta);
    float m2 = m * m;
    float m5 = m2 * m2 * m;

    return F0 + (make_float3(1.f) - F0) * m5;
}

__device__
BSDFVal Material::getBSDF(
        const Ray &ray,
        const HitRecord &hit,
    curandState* rngStates) const
{
    float3 normal = normalize(hit.normal);
    float3 wo = normalize(-ray.direction);
    BSDFVal bsdf;
    switch (type)
    {
    case LAMBERT:
        bsdf.direction = samplingLambert(normal, rngStates);
        bsdf.pdf = pdfLambert(normal, bsdf.direction);
        bsdf.brdf = evaluateLambert();
        break;
    // EMISSIVE à traiter séparément dans l'intégrateur
    /*case EMISSIVE:
        return color * intensity;*/

    case METAL:
    {
        float3 F0 = lerp(make_float3(0.04f), color, metalness);
        bsdf.direction = samplingGGX(wo, normal, rngStates);
        bsdf.pdf = pdfGGX(normal, bsdf.direction, wo);
        bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);
        break;
    }
    case PLASTIC:
    {
        float3 F0 = make_float3(0.04f);
        float cosTheta = saturate(dot(normal, wo));
        float3 F = fresnelSchlick(cosTheta, F0);
        float specWeight = (F.x + F.y + F.z) / 3.f;
        float probaReflect = curand_uniform(rngStates);
        if(probaReflect < F.x){
            bsdf.direction = samplingGGX(wo, normal, rngStates);
            bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);
        }
        else{
            bsdf.direction = samplingLambert(normal, rngStates);
            bsdf.brdf = evaluateLambert();
        }
        bsdf.pdf = pdfGGX(normal, bsdf.direction, wo) + pdfLambert(normal, bsdf.direction);
        break;
    }
    };
    return bsdf;
};
