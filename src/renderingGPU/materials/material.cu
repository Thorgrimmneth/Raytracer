#include "material.cuh"

DEVICE void Material::createONB(const float3 &n, float3 &tangent, float3 &bitangent) const
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

DEVICE float3 Material::toWorld(const float3 &normal, const float3 direction) const
{
    float3 T, B;
    createONB(normal, T, B);
    return normalize(direction.x * T + direction.y * B + direction.z * normal);
}

// ------------------------------------------------------------
//  Lambert
// ------------------------------------------------------------

DEVICE inline float3 Material::evaluateLambert() const { return color() * GPUInvPIf; }

DEVICE float3 Material::samplingLambert(const float3 normal, RNG &rngStates) const
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

DEVICE float Material::pdfLambert(const float3 normal, const float3 direction) const
{
    float cosTheta = fmaxf(dot(normal, direction), 0.f);
    return cosTheta * GPUInvPIf;
}

// ------------------------------------------------------------
//  GGX / Cook-Torrance
// ------------------------------------------------------------

DEVICE float Material::computeD(const float3 &p_normal, const float3 &h, const float alphaSquared) const
{
    float NdotH = fmaxf(dot(p_normal, h), 0.f);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (alphaSquared - 1.f) + 1.f);
    denom = 1.f / GPUPIf * denom * denom;
    return alphaSquared * denom;
}

DEVICE float3 Material::computeF(const float3 &wo, const float3 &h, const float3 &F0) const
{
    float HdotV = clamp(dot(h, wo), 0.f, 1.f);
    return F0 + (make_float3(1.f) - F0) * powf(1.f - HdotV, 5.f);
}

DEVICE float Material::computeG1(const float &NdotV, const float alphaSquared) const
{
    if (NdotV <= 0.f)
        return 0.f;
    float NdotV2 = NdotV * NdotV;
    float tan2 = (1.f - NdotV2) / fmaxf(NdotV2, 1e-8f);

    return 2.f / (1.f + sqrtf(1.f + alphaSquared * tan2));
}

DEVICE float Material::computeG(const float &NdotV, const float &NdotL, const float alphaSquared) const
{

    return computeG1(NdotV, alphaSquared) * computeG1(NdotL, alphaSquared);
}

DEVICE float3 Material::evaluateGGX(const float3 &wo, const float3 &normal, const float3 &wi,
                                           const float3 &F0) const
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

DEVICE float3 Material::samplingGGX(const float3 &wo, const float3 &normal, RNG &rngStates) const
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

DEVICE float Material::pdfGGX(const float3 n, const float3 wi, const float3 wo) const
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

    return (G1 * D) / (4.f * NdotV);
}

// ============================================================
//  getBSDF helpers
// ============================================================

DEVICE BSDFVal Material::getMetalBSDF(const float3 &direction, const float3 &normal,
                                      RNG &rngStates) const
{
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

DEVICE BSDFVal Material::getLambertBSDF(const float3 &direction, const float3 &normal,
                                        RNG &rngStates) const
{
    BSDFVal bsdf;

    bsdf.direction = samplingLambert(normal, rngStates);
    bsdf.pdf = pdfLambert(normal, bsdf.direction);
    bsdf.brdf = evaluateLambert();

    return bsdf;
}

DEVICE BSDFVal Material::getPlasticBSDF(const float3 &direction, const float3 &normal,
                                        RNG &rngStates) const
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

DEVICE BSDFVal Material::getMirrorBSDF(const float3 &direction, const float3 &normal) const
{
    BSDFVal bsdf;

    bsdf.direction = reflect(direction, normal);
    bsdf.pdf = 1.f;
    bsdf.brdf = color();

    return bsdf;
}

DEVICE BSDFVal Material::getTransparentBSDF(const float3 &direction, const float3 &normal,
                                            RNG &rngStates, bool &isInside) const
{
    float3 wo = direction;
    BSDFVal bsdf;

    float3 n = normal;
    float cosI = dot(n, -direction);

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
        bsdf.direction = reflect(wo, n);
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
        bsdf.direction = reflect(wo, n);
        bsdf.pdf = reff;
        bsdf.brdf = make_float3(1.f);
    }
    else
    {
        float3 wi = eta * (wo) + (eta * cosI - cosT) * n;

        bsdf.direction = normalize(wi);
        bsdf.pdf = 1.f - reff;
        bsdf.brdf = color() * make_float3(eta * eta);

        isInside = !isInside;
    }


    return bsdf;
}

// ============================================================
//  getBSDF dispatcher
// ============================================================

DEVICE BSDFVal Material::getBSDF(const float3 &direction, const float3 &normal, RNG &rngStates,
                                 bool &isInside) const
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

DEVICE float3 Material::evalLambertBSDF() const { return evaluateLambert(); }

DEVICE float3 Material::evalMetalBSDF(const float3 &direction, const float3 &normal,
                                      const float3 &wi) const
{
    float3 wo = -direction;

    if (dot(normal, wi) <= 0.f)
        return make_float3(0.f);

    float3 F0 = lerp(make_float3(0.04f), color(), metalness());

    return evaluateGGX(wo, normal, wi, F0);
}

DEVICE float3 Material::evalPlasticBSDF(const float3 &direction, const float3 &normal,
                                        const float3 &wi) const
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

DEVICE float3 Material::evalBSDF(const float3 &direction, const float3 &normal,
                                 const float3 &wi) const
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

DEVICE float Material::lambertPDF(const float3 &direction, const float3 &normal,
                                  const float3 &wi) const
{

    return pdfLambert(normal, wi);
}

DEVICE float Material::metalPDF(const float3 &direction, const float3 &normal,
                                const float3 &wi) const
{
    float3 wo = -direction;

    if (dot(normal, wi) <= 0.f)
        return 0.f;

    return pdfGGX(normal, wi, wo);
}

DEVICE float Material::plasticPDF(const float3 &direction, const float3 &normal,
                                  const float3 &wi) const
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

DEVICE float Material::pdf(const float3 &direction, const float3 &normal, const float3 &wi) const
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