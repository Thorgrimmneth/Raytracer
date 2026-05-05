#include "cuda_material.cuh"

__device__
void Material::createONB(const float3& n, float3& tangent, float3& bitangent) const
{
    if (n.z < -0.999f) {
        tangent    = make_float3(0.0f, -1.0f,  0.0f);
        bitangent  = make_float3(-1.0f, 0.0f,  0.0f);
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
    tangent   = normalize(tangent);
    bitangent = normalize(bitangent);
}

__device__
float3 Material::toWorld(const float3& normal, const float3 direction) const
{
    float3 T, B;
    createONB(normal, T, B);
    return normalize(direction.x * T + direction.y * B + direction.z * normal);
}

// ------------------------------------------------------------
//  Lambert
// ------------------------------------------------------------

__device__
inline float3 Material::evaluateLambert() const
{
    return color * GPUInvPIf;
}

__device__
float3 Material::samplingLambert(const float3 normal, curandState* rngStates) const
{
    float e1  = curand_uniform(rngStates);
    float e2  = curand_uniform(rngStates);
    float r   = sqrtf(e1);
    float phi = 2.f * GPUPIf * e2;
    float x   = r * cosf(phi);
    float y   = r * sinf(phi);
    float z   = sqrtf(fmaxf(0.f, 1.f - x*x - y*y));
    return toWorld(normal, make_float3(x, y, z));
}

__device__
float Material::pdfLambert(const float3 normal, const float3 direction) const
{
    float cosTheta = fmaxf(dot(normal, direction), 0.f);
    return cosTheta * GPUInvPIf;
}

// ------------------------------------------------------------
//  GGX / Cook-Torrance
// ------------------------------------------------------------

__device__
float Material::computeD(const float3& p_normal, const float3& h) const
{
    float a          = fmaxf(alpha, 1e-4f);
    float a2         = a * a;
    float NdotH      = fmaxf(dot(p_normal, h), 0.f);
    float NdotH2     = NdotH * NdotH;
    float denom      = GPUPIf * ((NdotH2 * (a2 - 1.f) + 1.f) * (NdotH2 * (a2 - 1.f) + 1.f));
    return a2 / fmaxf(denom, 1e-8f);
}

__device__
float3 Material::computeF(const float3& wo, const float3& h, const float3& F0) const
{
    float HdotV = clamp(dot(h, wo), 0.f, 1.f);
    return F0 + (make_float3(1.f) - F0) * powf(1.f - HdotV, 5.f);
}

__device__
float Material::computeG1(const float& x, const float& k) const
{
    return x / fmaxf(x * (1.f - k) + k, 1e-8f);
}

__device__
float Material::computeG(const float3& wi, const float3& wo, const float3& p_normal) const
{
    float k = ((ruggedness + 1.f) * (ruggedness + 1.f)) / 8.f;
    return computeG1(fmaxf(dot(p_normal, wo), 0.f), k)
         * computeG1(fmaxf(dot(p_normal, wi), 0.f), k);
}

__device__
inline float3 Material::evaluateGGX(
    const float3& wo, const float3& normal,
    const float3& wi, const float3& F0) const
{
    float3 h    = normalize(wi + wo);
    float NdotV = fmaxf(dot(normal, wo), 0.f);
    float NdotL = fmaxf(dot(normal, wi), 0.f);

    if (NdotV <= 0.f || NdotL <= 0.f)
        return make_float3(0.f);

    float  D    = computeD(normal, h);
    float3 F    = computeF(wo, h, F0);
    float  G    = computeG(wi, wo, normal);

    float denom = 4.f * NdotV * NdotL;
    if (denom < 1e-6f)
        return make_float3(0.f);

    return (D * G / denom) * F;
}

__device__
float3 Material::samplingGGX(
    const float3& wo, const float3& normal,
    curandState* rngStates) const
{
    float a = fmaxf(alpha, 1e-4f);

    float3 T, B;
    createONB(normal, T, B);

    float3 V = normalize(make_float3(dot(wo, T), dot(wo, B), dot(wo, normal)));

    V = normalize(make_float3(a * V.x, a * V.y, V.z));

    float3 T1 = (V.z < 0.9999f)
        ? normalize(cross(make_float3(0.f, 0.f, 1.f), V))
        : make_float3(1.f, 0.f, 0.f);
    float3 T2 = cross(V, T1);

    float e1  = curand_uniform(rngStates);
    float e2  = curand_uniform(rngStates);
    float r   = sqrtf(e1);
    float phi = 2.f * GPUPIf * e2;

    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);

    float s = 0.5f * (1.f + V.z);
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1*t1)) + s * t2;

    float3 Nh = t1*T1 + t2*T2 + sqrtf(fmaxf(0.f, 1.f - t1*t1 - t2*t2)) * V;

    float3 h = normalize(make_float3(a * Nh.x, a * Nh.y, fmaxf(0.f, Nh.z)));
    h = toWorld(normal, h);

    float3 wi = reflect(-wo, h);

    if (dot(normal, wi) <= 0.f)
        return make_float3(0.f);

    return wi;
}

__device__
float Material::pdfGGX(
    const float3 normal, const float3 wi, const float3 wo) const
{
    float3 h    = normalize(wo + wi);
    float  D    = computeD(normal, h);
    float  denom = 4.f * fmaxf(dot(wo, h), 1e-6f);
    return (D * fmaxf(dot(normal, h), 0.f)) / denom;
}

// ------------------------------------------------------------
//  Fresnel (Schlick)
// ------------------------------------------------------------

__device__
float3 Material::fresnelSchlick(float cosTheta, const float3& F0) const
{
    float m  = saturate(1.f - fabsf(cosTheta));
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return F0 + (make_float3(1.f) - F0) * m5;
}

// ------------------------------------------------------------
//  getBSDF
// ------------------------------------------------------------

__device__
BSDFVal Material::getBSDF(
    const Ray&      ray,
    const HitRecord& hit,
    curandState*    rngStates,
    bool&           isInside,
    bool&           reflected) const
{
    float3   normal = normalize(hit.normal);
    float3   wo     = normalize(-ray.direction);
    BSDFVal  bsdf;

    switch (type)
    {
    // ---- Lambertian ----
    case LAMBERT:
        bsdf.direction = samplingLambert(normal, rngStates);
        bsdf.pdf       = pdfLambert(normal, bsdf.direction);
        bsdf.brdf      = evaluateLambert();
        break;

    // ---- Metal (GGX) ----
    case METAL:
    {
        float3 F0      = lerp(make_float3(0.04f), color, metalness);
        bsdf.direction = samplingGGX(wo, normal, rngStates);

        if (dot(normal, bsdf.direction) <= 0.f) {
            bsdf.pdf  = 0.f;
            bsdf.brdf = make_float3(0.f);
            break;
        }

        bsdf.pdf  = pdfGGX(normal, bsdf.direction, wo);
        bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);
        break;
    }

    // ---- Plastic (diffuse + specular mix) ----
    case PLASTIC:
    {
        float3 F0       = make_float3(0.04f);
        float  cosTheta = saturate(dot(normal, wo));
        float3 F        = fresnelSchlick(cosTheta, F0);
        float  specW    = (F.x + F.y + F.z) / 3.f;

        if (curand_uniform(rngStates) < specW) {
            bsdf.direction = samplingGGX(wo, normal, rngStates);
            if (dot(normal, bsdf.direction) <= 0.f) {
                bsdf.pdf  = 0.f;
                bsdf.brdf = make_float3(0.f);
                break;
            }
            bsdf.brdf = evaluateGGX(wo, normal, bsdf.direction, F0);
        } else {
            bsdf.direction = samplingLambert(normal, rngStates);
            bsdf.brdf      = evaluateLambert();
        }

        bsdf.pdf = specW       * pdfGGX(normal, bsdf.direction, wo)
                 + (1.f - specW) * pdfLambert(normal, bsdf.direction);
        break;
    }

    // ---- Mirror (perfect specular) ----
    case MIRROR:
    {
        bsdf.direction = reflect(-wo, normal);
        bsdf.pdf       = 1.f;
        bsdf.brdf      = color;
        reflected      = true;
        break;
    }

    // ---- Transparent (Fresnel dielectric) ----
    case TRANSPARENT:
    {
        float3 n    = normal;
        float  cosI = dot(n, wo);
        if (cosI < 0.f) {
            n    = -n;
            cosI = -cosI;
        }

        float n1  = isInside ? ior : 1.f;
        float n2  = isInside ? 1.f : ior;
        float eta = n1 / n2;

        float k = 1.f - eta * eta * (1.f - cosI * cosI);

        // Total internal reflection
        if (k < 0.f) {
            bsdf.direction = reflect(-wo, n);
            bsdf.pdf       = 1.f;
            bsdf.brdf      = make_float3(1.f);
            reflected      = true;
            break;
        }

        float cosT = sqrtf(k);

        float rs = ((n1 * cosI) - (n2 * cosT)) /
                   ((n1 * cosI) + (n2 * cosT));
        rs *= rs;

        float rp = ((n2 * cosI) - (n1 * cosT)) /
                   ((n2 * cosI) + (n1 * cosT));
        rp *= rp;

        float reff = 0.5f * (rs + rp);
        float xi   = curand_uniform(rngStates);

        if (xi < reff) {
            // Reflection branch
            bsdf.direction = reflect(-wo, n);
            bsdf.pdf       = reff;
            bsdf.brdf      = make_float3(1.f);
            reflected      = true;
        } else {
            // Refraction branch
            float3 wi      = eta * (-wo) + (eta * cosI - cosT) * n;
            bsdf.direction = normalize(wi);
            bsdf.pdf       = 1.f - reff;
            float etaSq    = eta * eta;
            bsdf.brdf      = make_float3(etaSq);
            isInside       = !isInside;
            reflected      = false;
        }
        break;
    }
    } // switch

    return bsdf;
}

// ------------------------------------------------------------
//  evalBSDF  (used by NEE / MIS — delta materials return 0)
// ------------------------------------------------------------

__device__
float3 Material::evalBSDF(
    const Ray&       ray,
    const HitRecord& hit,
    const float3&    wi) const
{
    float3 normal = normalize(hit.normal);
    float3 wo     = normalize(-ray.direction);

    switch (type)
    {
    case LAMBERT:
        return evaluateLambert();

    case METAL:
    {
        float3 F0 = lerp(make_float3(0.04f), color, metalness);
        return evaluateGGX(wo, normal, wi, F0);
    }

    case PLASTIC:
    {
        float3 F0      = make_float3(0.04f);
        float  cosTheta = saturate(dot(normal, wo));
        float3 F       = fresnelSchlick(cosTheta, F0);
        float  specW   = (F.x + F.y + F.z) / 3.f;

        float3 diffuse  = evaluateLambert();
        float3 specular = evaluateGGX(wo, normal, wi, F0);

        return (1.f - specW) * diffuse + specW * specular;
    }

    // Delta materials have no well-defined BSDF for NEE
    case MIRROR:
    case TRANSPARENT:
    default:
        return make_float3(0.f);
    }
}

// ------------------------------------------------------------
//  pdf  (used by MIS — delta materials return 0)
// ------------------------------------------------------------

__device__
float Material::pdf(
    const Ray&       ray,
    const HitRecord& hit,
    const float3&    wi) const
{
    float3 normal = normalize(hit.normal);
    float3 wo     = normalize(-ray.direction);

    switch (type)
    {
    case LAMBERT:
        return pdfLambert(normal, wi);

    case METAL:
        return pdfGGX(normal, wi, wo);

    case PLASTIC:
    {
        float3 F0      = make_float3(0.04f);
        float  cosTheta = saturate(dot(normal, wo));
        float3 F       = fresnelSchlick(cosTheta, F0);
        float  specW   = (F.x + F.y + F.z) / 3.f;

        return specW         * pdfGGX(normal, wi, wo)
             + (1.f - specW) * pdfLambert(normal, wi);
    }

    default:
        return 0.f;
    }
}
