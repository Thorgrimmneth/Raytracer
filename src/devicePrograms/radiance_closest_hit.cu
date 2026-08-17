#include "../renderingGPU/lights/light.cuh"
#include "../renderingGPU/lights/light_selection_utils.cuh"
#include "../renderingGPU/lights/lightsample.cuh"
#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/optix/optix_sbt_manager.h"
#include "../renderingGPU/renderingUtils/shading_kernels.cuh"
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
            case MaterialType::EMISSIVE:
            {
                payload->contribution = mtl.color() * mtl.intensity();
                payload->bsdfDir = make_float3(0.f);
                payload->bsdfPdf = 0.f;
                return;
            }

            case MaterialType::LAMBERT:
            {
                // NEE for diffuse
                if (!payload->lastBounceWasDelta && params.nbLights > 0)
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
                                      OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                      1, 1, 1, p0, p1);

                            nee_contribution = params.throughputs[qid] * f_nee * ls.radiance 
                                             * cosTheta_nee * w * (1.f / lightPdf) * shadowPayload.transmittance;
                        }
                    }
                }

                // BSDF sampling
                bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
                break;
            }

            case MaterialType::METAL:
            {
                // No NEE for metals, only BSDF sampling
                bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
                break;
            }

            case MaterialType::PLASTIC:
            {
                // NEE for plastic (has diffuse component)
                if (!payload->lastBounceWasDelta && params.nbLights > 0)
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
                                      OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                      1, 1, 1, p0, p1);

                            nee_contribution = params.throughputs[qid] * f_nee * ls.radiance 
                                             * cosTheta_nee * w * (1.f / lightPdf) * shadowPayload.transmittance;
                        }
                    }
                }

                // BSDF sampling
                bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
                break;
            }

            case MaterialType::TRANSPARENT:
            {
                // No NEE for transparent (delta material), only BSDF sampling
                int tempIsInside = payload->isInside;
                bsdf = mtl.getBSDF(wo, N, rng, tempIsInside);
                payload->isInside = tempIsInside;
                break;
            }

            case MaterialType::MIRROR:
            {
                // No NEE for mirrors (delta material), only BSDF sampling
                bsdf = mtl.getBSDF(wo, N, rng, payload->isInside);
                break;
            }

            default:
            {
                payload->bsdfDir = make_float3(0.f);
                payload->bsdfPdf = 0.f;
                payload->contribution = make_float3(0.f);
                return;
            }
        }

        // ============================================================
        // Finalize bounce
        // ============================================================

        if (bsdf.pdf <= 0.f || dot(N, bsdf.direction) <= 0.f)
        {
            payload->bsdfDir = make_float3(0.f);
            payload->bsdfPdf = 0.f;
            payload->contribution = nee_contribution;
            return;
        }

        float cosTheta_bsdf = dot(N, bsdf.direction);
        float3 next_throughput = params.throughputs[qid] * bsdf.brdf * cosTheta_bsdf / bsdf.pdf;

        payload->bsdfDir = bsdf.direction;
        payload->bsdfPdf = bsdf.pdf;
        payload->lastBounceWasDelta = (matType == MaterialType::MIRROR || matType == MaterialType::TRANSPARENT) ? 1 : 0;
        
        params.throughputs[qid] = next_throughput;
        payload->contribution = nee_contribution;
    }
}