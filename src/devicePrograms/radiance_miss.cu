#include "../renderingGPU/optix/optix_payload.h"
#include "../renderingGPU/utils/op.cuh"
#include "../renderingGPU/utils/packing.h"
#include "launch_radiance_params.cuh"
#include <optix.h>
#include <optix_device.h>

extern "C" {
__constant__ LaunchRadianceParams params;
}

extern "C" __global__ void __miss__radiance()
{
    Payload *payload = reinterpret_cast<Payload *>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
    //payload->hit = 0;
    float3 direction = optixGetWorldRayDirection();
    if (direction.y <= -0.3f)
    {
        payload->accumulated_color = make_float3(0.f);
        payload->hit = 0;
        return;
    }

    const float horizonFade = smoothstep(-0.2f, 0.1f, direction.y);

    const float segmentLength = params.atmosphereSize / params.nbSkySamples;

    //------------------------------------------------------------------
    // Phases
    //------------------------------------------------------------------
    float3 SUN_DIRECTION = params.sunDirection;
    const float mu = dot(direction, SUN_DIRECTION);
    const float mu2Term = 1.0f + mu * mu;

    const float phaseR = 0.0596831f * mu2Term;

    const float temp = 1.5776f - 1.52f * mu;

    const float phaseM = 0.0195609427f * mu2Term * rsqrtf(temp) / temp;

    //------------------------------------------------------------------
    // Soleil : somme géométrique fermée
    //------------------------------------------------------------------

    const float sunSegmentLength = 15000.0f;
    const float qR = __expf(-params.HR * SUN_DIRECTION.y * sunSegmentLength);

    const float qM = __expf(-params.HM * SUN_DIRECTION.y * sunSegmentLength);

    const float invOneMinusQR = 1.0f / (1.0f - qR);

    const float invOneMinusQM = 1.0f / (1.0f - qM);

    float qR2 = qR * qR;
    float qR4 = qR2 * qR2;

    float qM2 = qM * qM;
    float qM4 = qM2 * qM2;

    const float geoFactorR = sunSegmentLength * (1.0f - qR4) * invOneMinusQR;

    const float geoFactorM = sunSegmentLength * (1.0f - qM4) * invOneMinusQM;

    //------------------------------------------------------------------
    // Intégration principale
    //------------------------------------------------------------------

    float3 sumR = make_float3(0.0f);
    float3 sumM = make_float3(0.0f);

    float opticalDepthR = 0.0f;
    float opticalDepthM = 0.0f;

    const float3 origin = optixGetWorldRayOrigin();
    float3 samplePosition = origin + direction * (0.5f * segmentLength);

    float height = fmaxf(samplePosition.y, 0.0f);

    float hrLocal = __expf(-height * params.HR);

    float hmLocal = __expf(-height * params.HM);

    const float rR = __expf(-params.HR * direction.y * segmentLength);

    const float rM = __expf(-params.HM * direction.y * segmentLength);

    for (int i = 0; i < params.nbSkySamples; ++i)
    {
        opticalDepthR = fmaf(hrLocal, segmentLength, opticalDepthR);

        opticalDepthM = fmaf(hmLocal, segmentLength, opticalDepthM);

        //--------------------------------------------------------------
        // Profondeur optique vers le soleil
        //--------------------------------------------------------------

        const float firstR = hrLocal * qR;

        const float firstM = hmLocal * qM;

        const float opticalDepthLightR = firstR * geoFactorR;

        const float opticalDepthLightM = firstM * geoFactorM;

        //--------------------------------------------------------------
        // Atténuation
        //--------------------------------------------------------------

        const float3 tau =
            -(params.betaR * (opticalDepthR + opticalDepthLightR) + params.betaM * (opticalDepthM + opticalDepthLightM));

        const float3 attenuation = make_float3(__expf(tau.x), __expf(tau.y), __expf(tau.z));

        sumR = sumR + attenuation * (hrLocal * segmentLength);
        sumM = sumM + attenuation * (hmLocal * segmentLength);

        //--------------------------------------------------------------
        // Avance au prochain échantillon
        //--------------------------------------------------------------

        hrLocal = hrLocal * rR;
        hmLocal = hmLocal * rM;
    }
    //------------------------------------------------------------------
    // Couleur du ciel
    //------------------------------------------------------------------

    float3 sky = sumR * params.betaR * phaseR + sumM * params.betaM * phaseM * 0.3f;

    //------------------------------------------------------------------
    // Disque solaire
    //------------------------------------------------------------------

    const float cosTheta = dot(direction, params.sunDirection);

    const float sunDisk = smoothstep(params.sunAngularRadius, params.sunHalfAngularRadius, cosTheta);

    const float t = clamp((params.sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);

    const float sunset = (1.0f - t) * (1.0f - t);

    float3 sunColor = lerp(make_float3(30.f, 27.f, 24.f), make_float3(60.f, 25.f, 10.f), sunset);

    if (params.depth > 0)
    {
        sunColor = clamp(sunColor, make_float3(0.f), make_float3(1.f));
    }

    sky = sky + sunColor * sunDisk;

    //------------------------------------------------------------------
    // Intensité globale
    //------------------------------------------------------------------

    const float mult = 1.0f + 19.0f * t * t * t;

    sky = sky * mult * horizonFade;

    //------------------------------------------------------------------
    // Accumulation
    //------------------------------------------------------------------
    payload->accumulated_color = params.throughputs[optixGetLaunchIndex().x] * sky;
    payload->hit = 0;
}