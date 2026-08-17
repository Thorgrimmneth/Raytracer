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
    payload->objectIndex = instanceIndex;
    payload->object_type = HIT_TRIANGLE_MESH;

    // ============================================================
    // Lambert Shading
    // ============================================================

    uint qid = optixGetLaunchIndex().x;

    if (qid < params.active_count && params.rngs)
    {
        const Material &mtl = params.lightContext.materials[payload->materialIndex];

        // Skip non-Lambert materials
        if (mtl.type() != MaterialType::LAMBERT)
            return;

        RNG &rng = params.rngs[params.pixelIndices[qid]];
        float3 direction = optixGetWorldRayDirection();

        // Lambert BSDF sampling
        float e1 = rng.nextFloat();
        float e2 = rng.nextFloat();
        float r = sqrtf(e1);
        float phi = 2.f * M_PIf * e2;
        float cosf_phi = cosf(phi);
        float sinf_phi = sinf(phi);
        float x = r * cosf_phi;
        float y = r * sinf_phi;
        float z = sqrtf(fmaxf(0.f, 1.f - x * x - y * y));

        float3 T, B;
        getTangentFrame(N, T, B);
        float3 bsdfDir = x * T + y * B + z * N;

        float cosTheta_bsdf = fmaxf(dot(N, bsdfDir), 0.f);
        float bsdf_pdf = cosTheta_bsdf * M_1_PIf;

        if (bsdf_pdf <= 0.f)
        {
            payload->luminous_contribution = make_float3(0.f);
            return;
        }

        if (params.nbLights > 0)
        {
            int lightIndex = selectLightByImportance(params.nbLights, params.lightCumulativeWeights, rng);

            Light &light = params.lights[lightIndex];

            //LightSample ls = light.sample(payload->position, rng, params.lightContext);
        }
        payload->bsdfDir = bsdfDir;
        payload->bsdfPdf = bsdf_pdf;

        // Store contribution for accumulation
        payload->luminous_contribution = mtl.color() * cosTheta_bsdf * M_1_PIf / bsdf_pdf;
    }
}