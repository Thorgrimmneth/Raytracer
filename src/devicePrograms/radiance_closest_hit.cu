#include "../renderingGPU/lights/light.cuh"
#include "../renderingGPU/lights/light_selection_utils.cuh"
#include "../renderingGPU/lights/lightsample.cuh"
#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"
#include "../renderingGPU/utils/rng.cuh"
#include "launch_radiance_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const uint primID = optixGetPrimitiveIndex();

    const uint3 tri = data->mesh.triangles[primID];

    const float2 bc = optixGetTriangleBarycentrics();

    const float3 &n0 = data->mesh.normals[tri.x];
    const float3 &n1 = data->mesh.normals[tri.y];
    const float3 &n2 = data->mesh.normals[tri.z];

    // Interpolate normal in model space
    float3 N = normalize((1 - bc.x - bc.y) * n0 + bc.x * n1 + bc.y * n2);

    // Get material index and apply mesh transformation to normal
    uint instanceIndex = optixGetInstanceId();

    if (params.lightContext.meshInstances && instanceIndex < params.nbMeshInstances)
    {
        // Transform normal from model space to world space
        const float *transform = params.lightContext.meshInstances[instanceIndex].transform;

        float3 N_world = transformVector(transform, N);

        N = normalize(N_world);
        payload->materialIndex = params.lightContext.meshInstances[instanceIndex].materialIndex;
    }
    else
    {
        payload->materialIndex = 0;
    }

    // Orient normal to face the incoming ray
    if (dot(N, optixGetWorldRayDirection()) > 0.f)
    {
        N = -N;
    }

    payload->hit = 1;

    payload->t = optixGetRayTmax();
    float3 position = optixGetWorldRayOrigin() + payload->t * optixGetWorldRayDirection();
    payload->position = position;

    payload->normal = N;

    // ============================================================
    // Material Shading - Switch by Material Type
    // ============================================================

    uint qid = optixGetLaunchIndex().x;

    if (qid < params.active_count && params.rngs)
    {
        const Material &mtl = params.lightContext.materials[payload->materialIndex];
        MaterialType matType = mtl.type();
        RNG &rng = params.rngs[qid];
        float3 wo = optixGetWorldRayDirection();
        float3 nee_contribution = make_float3(0.f);
        BSDFVal bsdf;

        switch (matType)
        {
        case MaterialType::EMISSIVE: {
            if (payload->lastBounceWasDelta == 1 || payload->depth == 0)
            {
                payload->contribution = mtl.color() * mtl.intensity() * params.throughputs[qid];
            }

            else
            {
                const MeshInstance &inst = params.lightContext.meshInstances[instanceIndex];

                float areaPdf = 1.f / inst.worldArea;

                float dist2 = payload->t * payload->t;
                float cosTheta = max(dot(N, -optixGetWorldRayDirection()), 0.f);
                if (cosTheta <= 0.f)
                {
                    return;
                }

                float lightPdf = getLightProbability(params.nbLights, params.lightProbabilities, inst.lightIndex) * areaPdf *
                           dist2 / cosTheta;

                float w = powerHeuristic(payload->lastBsdfPdf, lightPdf);

                payload->contribution += params.throughputs[qid] * mtl.color() * mtl.intensity() * w;
            }

            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            return;
        }

        case MaterialType::LAMBERT: {
            // NEE for diffuse
            if (params.nbLights > 0)
            {
                int lightIndex = selectLightByImportance(params.nbLights, params.lightCumulativeWeights, rng);
                Light &light = params.lights[lightIndex];
                LightSample ls = light.sample(payload->position, rng, params.lightContext);

                float cosTheta_nee = fmaxf(dot(N, ls.direction), 0.0f);
                float lightPdf = ls.pdf * getLightProbability(params.nbLights, params.lightProbabilities, lightIndex);

                if (lightPdf > 0.f && cosTheta_nee > 0.f && ls.pdf > 0.f)
                {
                    float3 f_nee = mtl.evalBSDF(wo, N, ls.direction);

                    if (dot(f_nee, f_nee) > 0.f)
                    {
                        float pdf_bsdf = mtl.pdf(wo, N, ls.direction);
                        float w = powerHeuristic(lightPdf, pdf_bsdf);

                        ShadowPayload shadowPayload;
                        shadowPayload.transmittance = make_float3(1.f);
                        shadowPayload.depth = 0;

                        uint32_t p0, p1;
                        packPointer(&shadowPayload, p0, p1);

                        float3 shadowRayOrigin = payload->position + N * 1e-3f;
                        float shadowRayDist = ls.distance - 1e-3f;

                        optixTrace(params.traversable, shadowRayOrigin, ls.direction, 1e-3f, shadowRayDist, 0.0f,
                                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW, p0, p1);

                        nee_contribution = params.throughputs[qid] * f_nee * ls.radiance * cosTheta_nee * w *
                                           (1.f / lightPdf) * shadowPayload.transmittance;
                    }
                }
            }

            // BSDF sampling
            bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
            break;
        }

        case MaterialType::METAL: {
            // No NEE for metals, only BSDF sampling
            bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
            break;
        }

        case MaterialType::PLASTIC: {
            // NEE for plastic (has diffuse component)
            if (params.nbLights > 0)
            {
                int lightIndex = selectLightByImportance(params.nbLights, params.lightCumulativeWeights, rng);
                Light &light = params.lights[lightIndex];
                LightSample ls = light.sample(payload->position, rng, params.lightContext);

                float cosTheta_nee = fmaxf(dot(N, ls.direction), 0.0f);
                float lightPdf = ls.pdf * getLightProbability(params.nbLights, params.lightProbabilities, lightIndex);

                if (lightPdf > 0.f && cosTheta_nee > 0.f && ls.pdf > 0.f)
                {
                    float3 f_nee = mtl.evalBSDF(wo, N, ls.direction);

                    if (dot(f_nee, f_nee) > 0.f)
                    {
                        float pdf_bsdf = mtl.pdf(wo, N, ls.direction);
                        float w = powerHeuristic(lightPdf, pdf_bsdf);

                        ShadowPayload shadowPayload;
                        shadowPayload.transmittance = make_float3(1.f);
                        shadowPayload.depth = 0;

                        uint32_t p0, p1;
                        packPointer(&shadowPayload, p0, p1);

                        float3 shadowRayOrigin = payload->position + N * 1e-3f;
                        float shadowRayDist = ls.distance - 1e-3f;

                        optixTrace(params.traversable, shadowRayOrigin, ls.direction, 1e-3f, shadowRayDist, 0.0f,
                                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW, p0, p1);

                        nee_contribution = params.throughputs[qid] * f_nee * ls.radiance * cosTheta_nee * w *
                                           (1.f / lightPdf) * shadowPayload.transmittance;
                    }
                }
            }

            // BSDF sampling
            bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
            break;
        }

        case MaterialType::TRANSPARENT: {
            // No NEE for transparent (delta material), only BSDF sampling
            int tempIsInside = payload->isInside;
            bsdf = mtl.getBSDF(wo, N, rng, tempIsInside);
            payload->isInside = tempIsInside;
            break;
        }

        case MaterialType::MIRROR: {
            // No NEE for mirrors (delta material), only BSDF sampling
            bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
            break;
        }

        default: {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->contribution = make_float3(0.f);
            return;
        }
        }

        // ============================================================
        // Finalize bounce
        // ============================================================

        bool isDelta = matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT;

        if (bsdf.pdf <= 0.f)
        {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->contribution = nee_contribution;
            return;
        }

        // Le test d'hémisphère ne s'applique PAS à la transmission.
        if (!isDelta && dot(N, bsdf.direction) <= 0.f)
        {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->contribution = nee_contribution;
            return;
        }

        float cosTheta_bsdf = dot(N, bsdf.direction);
        float3 next_throughput = params.throughputs[qid] * bsdf.brdf;
        if (!(matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT))
        {
            next_throughput = next_throughput * cosTheta_bsdf / bsdf.pdf;
        }

        payload->bsdfDir = bsdf.direction;
        payload->bsdfPdf = bsdf.pdf;
        payload->lastBounceWasDelta = (matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT) ? 1 : 0;

        params.throughputs[qid] = next_throughput;
        payload->contribution = nee_contribution;
    }
}

__forceinline__ __device__ float pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

__forceinline__ __device__ void getTangentFrame(float3 normal, float3 &T, float3 &B)
{
    float nx = normal.x;
    float ny = normal.y;
    float sign = copysignf(1.0f, normal.z);
    float a = -1.0f / (sign + normal.z);
    float b = nx * ny * a;
    T = make_float3(1.0f + sign * nx * nx * a, sign * b, -sign * nx);
    B = make_float3(b, sign + ny * ny * a, -ny);
}

__forceinline__ __device__ float3 sampleGGX(float u1, float u2, float Vx, float Vy, float Vz)
{
    // GGX sampling Disney/Burley
    float r = sqrtf(u1);
    float phi = 2.f * M_PIf * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    float t1 = r * cosPhi;
    float t2 = r * sinPhi;
    float s = 0.5f * (1.f + Vz);
    float t1S = t1 * t1;
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1S)) + s * t2;
    float t2S = t2 * t2;
    float lenNh = sqrtf(fmaxf(1e-8f, t1S + t2S + (1.f - t1S - t2S)));
    return make_float3(t1 / lenNh, t2 / lenNh, sqrtf(fmaxf(0.f, 1.f - t1S - t2S)) / lenNh);
}
