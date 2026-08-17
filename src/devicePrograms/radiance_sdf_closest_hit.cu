#include "../renderingGPU/lights/light.cuh"
#include "../renderingGPU/lights/light_selection_utils.cuh"
#include "../renderingGPU/lights/lightsample.cuh"
#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/utils/object_type.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"
#include "launch_radiance_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __closesthit__radiance__sdf()
{
    const HitData *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const uint primIdx = optixGetPrimitiveIndex();

    const SDF &sdf = data->sdf.sdfs[primIdx];

    float3 p = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float3 NWorld;

    if (sdf.type == SDFType::SphereAnalytic)
    {
        NWorld = sdf.sphere.getNormal(p, sdf.translation);
    }
    else
    {
        float3 pLocal = transform(sdf.rotation, p - sdf.translation);
        float3 N = sdf.getNormal(pLocal);

        NWorld = transform(sdf.rotation.transpose(), N);
    }

    NWorld = normalize(NWorld);

    // Orienter la normale contre le rayon incident
    if (dot(NWorld, optixGetWorldRayDirection()) > 0.f)
    {
        NWorld = -NWorld;
    }

    payload->hit = 1;
    payload->t = optixGetRayTmax();
    payload->position = p;
    payload->normal = NWorld;
    payload->objectIndex = optixGetInstanceId();
    payload->materialIndex = sdf.getMaterialIndex();
    payload->object_type = HIT_SDF;

    // ============================================================
    // Material Shading - All Materials
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
            payload->luminous_contribution = params.throughputs[qid] * mtl.color() * mtl.intensity();
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            return;
        }

        case MaterialType::LAMBERT: {
            // NEE for diffuse
            if (!payload->lastBounceWasDelta && params.nbLights > 0)
            {
                int lightIndex = selectLightByImportance(params.nbLights, params.lightCumulativeWeights, rng);
                Light &light = params.lights[lightIndex];
                LightSample ls = light.sample(payload->position, rng, params.lightContext);

                float cosTheta_nee = fmaxf(dot(NWorld, ls.direction), 0.0f);
                float lightPdf = ls.pdf * getLightProbability(params.nbLights, params.lightProbabilities, lightIndex);

                if (lightPdf > 0.f && cosTheta_nee > 0.f && ls.pdf > 0.f)
                {
                    float3 f_nee = mtl.evalBSDF(wo, NWorld, ls.direction);

                    if (dot(f_nee, f_nee) > 0.f)
                    {
                        float pdf_bsdf = mtl.pdf(wo, NWorld, ls.direction);
                        float w = powerHeuristic(lightPdf, pdf_bsdf);

                        ShadowPayload shadowPayload;
                        shadowPayload.transmittance = make_float3(1.f);
                        shadowPayload.depth = 0;

                        uint32_t p0, p1;
                        packPointer(&shadowPayload, p0, p1);

                        float3 shadowRayOrigin = payload->position + NWorld * 1e-3f;
                        float shadowRayDist = ls.distance - 1e-3f;

                        optixTrace(params.traversable, shadowRayOrigin, ls.direction, 1e-3f, shadowRayDist, 0.0f,
                                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 1, 1, 1, p0, p1);

                        nee_contribution = params.throughputs[qid] * f_nee * ls.radiance * cosTheta_nee * w *
                                           (1.f / lightPdf) * shadowPayload.transmittance;
                    }
                }
            }

            // BSDF sampling
            bsdf = mtl.getBSDF(wo, NWorld, rng, payload->isInside);
            break;
        }

        case MaterialType::METAL: {
            // No NEE for metals, only BSDF sampling
            bsdf = mtl.getBSDF(wo, NWorld, rng, payload->isInside);
            break;
        }

        case MaterialType::PLASTIC: {
            // NEE for plastic (has diffuse component)
            if (!payload->lastBounceWasDelta && params.nbLights > 0)
            {
                int lightIndex = selectLightByImportance(params.nbLights, params.lightCumulativeWeights, rng);
                Light &light = params.lights[lightIndex];
                LightSample ls = light.sample(payload->position, rng, params.lightContext);

                float cosTheta_nee = fmaxf(dot(NWorld, ls.direction), 0.0f);
                float lightPdf = ls.pdf * getLightProbability(params.nbLights, params.lightProbabilities, lightIndex);

                if (lightPdf > 0.f && cosTheta_nee > 0.f && ls.pdf > 0.f)
                {
                    float3 f_nee = mtl.evalBSDF(wo, NWorld, ls.direction);

                    if (dot(f_nee, f_nee) > 0.f)
                    {
                        float pdf_bsdf = mtl.pdf(wo, NWorld, ls.direction);
                        float w = powerHeuristic(lightPdf, pdf_bsdf);

                        ShadowPayload shadowPayload;
                        shadowPayload.transmittance = make_float3(1.f);
                        shadowPayload.depth = 0;

                        uint32_t p0, p1;
                        packPointer(&shadowPayload, p0, p1);

                        float3 shadowRayOrigin = payload->position + NWorld * 1e-3f;
                        float shadowRayDist = ls.distance - 1e-3f;

                        optixTrace(params.traversable, shadowRayOrigin, ls.direction, 1e-3f, shadowRayDist, 0.0f,
                                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 1, 1, 1, p0, p1);

                        nee_contribution = params.throughputs[qid] * f_nee * ls.radiance * cosTheta_nee * w *
                                           (1.f / lightPdf) * shadowPayload.transmittance;
                    }
                }
            }

            // BSDF sampling
            bsdf = mtl.getBSDF(wo, NWorld, rng, payload->isInside);
            break;
        }

        case MaterialType::TRANSPARENT: {
            // No NEE for transparent (delta material), only BSDF sampling
            int tempIsInside = payload->isInside;
            bsdf = mtl.getBSDF(wo, NWorld, rng, tempIsInside);
            payload->isInside = tempIsInside;
            break;
        }

        case MaterialType::MIRROR: {
            // No NEE for mirrors (delta material), only BSDF sampling
            bsdf = mtl.getBSDF(wo, NWorld, rng, payload->isInside);
            break;
        }

        default: {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->luminous_contribution = make_float3(0.f);
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
            payload->luminous_contribution = nee_contribution;
            return;
        }

        // Le test d'hémisphère ne s'applique PAS à la transmission.
        if (!isDelta && dot(NWorld, bsdf.direction) <= 0.f)
        {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->luminous_contribution = nee_contribution;
            return;
        }

        float cosTheta_bsdf = dot(NWorld, bsdf.direction);
        float3 next_throughput = params.throughputs[qid] * bsdf.brdf;
        if (!(matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT))
        {
            next_throughput = next_throughput * cosTheta_bsdf / bsdf.pdf;
        }

        payload->bsdfDir = bsdf.direction;
        payload->bsdfPdf = bsdf.pdf;
        payload->lastBounceWasDelta = (matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT) ? 1 : 0;

        params.throughputs[qid] = next_throughput;
        payload->luminous_contribution = nee_contribution;
    }
}