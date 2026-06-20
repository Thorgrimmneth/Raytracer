#include "pathtracer_integrator.cuh"

#include "../scene/scene.cuh"
#include "../lights/light.cuh"



DEVICE 
float3 PathtracerIntegrator::lighting(const CudaScene &scene, const float3 &origin, const float3 &direction, const float tMin,
                                             const float tMax, RNG &rng)
{
    float3 finalColor = make_float3(0.f);

    float3 primOrigin = origin;
    float3 primDirection = direction;
    float3 throughput = make_float3(1.f);
    bool isInside = false;
    bool lastBounceWasDelta = true;
    float lastBsdfPdf = 1.f;
    return finalColor;
    /*for (int depth = 0; depth < nbBounces; depth++)
    {
        OptixHit hit;
        if (!scene.intersect(primOrigin, primDirection, tMin, tMax, hit))
        {
            finalColor += throughput * getSkyColor(primOrigin, primDirection, depth == 0);
            break;
        }
        // return finalColor;
        const Material &mtl = scene.materials[hit.materialIndex];
        if (mtl.type() == MaterialType::EMISSIVE)
        {
            float3 emission = mtl.color() * mtl.intensity();

            if (lastBounceWasDelta)
            {
                finalColor += throughput * emission;
            }
            else
            {
                float lightPdf = scene.lightPdf(primOrigin, primDirection);

                float w = powerHeuristic(lastBsdfPdf, lightPdf);

                finalColor += throughput * emission * w;
            }

            break;
        }

        BSDFVal bsdf = mtl.getBSDF(primDirection, hit.normal, rng, isInside);
        if (bsdf.pdf <= 1e-4f)
            break;
        if (mtl.type() == MaterialType::MIRROR)
        {
            throughput *= bsdf.brdf;
        }
        else
        {
            // Select light by importance (weighted by intensity)
            int lightIndex = selectLightByImportance(scene, rng);
            float lightSelectionProb = getLightProbability(scene.nbLights, scene.lightProbabilities, lightIndex);

            const Light &light = scene.lights[lightIndex];
            LightSample ls = light.sample(hit.position, rng, scene);

            if (ls.pdf > 0.f)
            {

                float3 shadowTint = scene.traceShadowRay(hit.position + hit.normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);
                
                // If shadow ray wasn't completely blocked
                if (length(shadowTint) > 1e-6f)
                {
                    float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                    if (cosTheta > 0.f)
                    {
                        float3 f = mtl.evalBSDF(primDirection, hit.normal, ls.direction);

                        float pdf_light = ls.pdf * lightSelectionProb;

                        float pdf_bsdf = mtl.pdf(primDirection, hit.normal, ls.direction);

                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        finalColor += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
            float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

            throughput = throughput * bsdf.brdf * cosTheta / bsdf.pdf;
        }
        lastBounceWasDelta = mtl.type() == MaterialType::MIRROR;
        lastBsdfPdf = bsdf.pdf;
        if (depth > 2)
        {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = clamp(p, 0.1f, 1.f);

            if (rng.nextFloat() > p)
                break;

            throughput /= p;
        }
        primOrigin = hit.position + bsdf.direction * 1e-3f;
        primDirection = bsdf.direction;
    }

    return finalColor;*/
}

DEVICE 
float3 PathtracerIntegrator::getSkyColor(const float3 &origin, const float3 &direction, bool safeSun)
{
    float3 rayDir = direction;
    // float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);
    float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);

    float mult = 1.0f + 19.0f * t * t * t;

    const float horizonFade = smoothstep(-0.05f, 0.02f, direction.y);

    if (horizonFade <= 0.0f)
    {
        return make_float3(0.0f);
    }

    float segmentLength = sizeAtmosphere / skyColorSamples;
    float tCurrent = 0.0f;

    float3 sumR = make_float3(0.f);
    float3 sumM = make_float3(0.f);

    float opticalDepthR = 0.0f;
    float opticalDepthM = 0.0f;

    float mu = dot(direction, sunDirection);

    //float g = 0.76f;

    float mu2Term = 1.0f + mu * mu;

    float phaseR = 0.0596831f * mu2Term; 
    //float phaseR = (3.0f / (16.0f * GPUPIf)) * (1.0f + mu * mu);

    float temp = 1.5776f - 1.52f * mu; 
    //float temp = 1.0f + g * g - 2.0f * g * mu;

    //float phaseM = (3.0f / (8.0f * GPUPIf)) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * temp * sqrtf(temp));
    float phaseM = 0.0195609427f * mu2Term * rsqrtf(temp) / temp;

    const int sunSamples = 4;
    float sunSegmentLength = 15000.f;

    for (int i = 0; i < skyColorSamples; ++i)
    {
        float3 samplePosition = origin + rayDir * (tCurrent + segmentLength * 0.5f);

        float height = fmaxf(samplePosition.y, 0.0f);

        float hrLocal = __expf(-height / hr);
        float hmLocal = __expf(-height / hm);

        opticalDepthR += hrLocal * segmentLength;
        opticalDepthM += hmLocal * segmentLength;

        float3 sunSamplePosition = samplePosition;

        float opticalDepthLightR = 0.0f;
        float opticalDepthLightM = 0.0f;

        for (int j = 0; j < sunSamples; ++j)
        {
            sunSamplePosition += sunDirection * sunSegmentLength;

            float heightLight = fmaxf(sunSamplePosition.y, 0.0f);

            opticalDepthLightR += __expf(-heightLight / hr) * sunSegmentLength;
            opticalDepthLightM += __expf(-heightLight / hm) * sunSegmentLength;
        }

        float3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * (opticalDepthM + opticalDepthLightM);

        float3 attenuation = make_float3(__expf(-tau.x), __expf(-tau.y), __expf(-tau.z));

        sumR += attenuation * hrLocal * segmentLength;
        sumM += attenuation * hmLocal * segmentLength;

        tCurrent += segmentLength;
    }

    float3 sky = sumR * betaR * phaseR + sumM * betaM * phaseM * 0.3f;

    float sunAngularRadius = 2.1f * GPUPIf / 180.f;
    float cosTheta = dot(direction, sunDirection);

    float sunDisk =
        smoothstep(cos(sunAngularRadius),
                   cos(sunAngularRadius * 0.5f),
                   cosTheta);

    float sunset = pow(1.0f - t, 2.0f);
    float3 sunColor =
        lerp(make_float3(30.f,27.f,24.f),
            make_float3(60.f,25.f,10.f),
            sunset);
    if(!safeSun){
        sunColor = clamp(sunColor, make_float3(0.f), make_float3(1.f));
    }
    sky += sunColor * sunDisk;

    return sky * mult * horizonFade;
}