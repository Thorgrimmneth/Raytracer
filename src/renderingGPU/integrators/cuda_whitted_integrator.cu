#include "../cuda_scene.cuh"
#include "../lights/cuda_light.cuh"
#include "cuda_whitted_integrator.cuh"

__device__
float3 WhittedIntegrator::lighting(
        const CudaScene &scene,
        const Ray &primaryRay,
        const float tMin,
        const float tMax,
        curandState *rng)
{
    float3 finalColor = make_float3(0.f);

    Ray ray = primaryRay;
    float3 throughput = make_float3(1.f);
    bool isInside = false;
    bool reflected;
    float lastPdf = 1.f;
    for (int depth = 0; depth < nbBounces; depth++)
    {
        reflected = true;
        HitRecord hit;

        if (!scene.intersect(ray, tMin, tMax, hit))
        {
            finalColor += throughput * getSkyColor(ray);
            break;
        }

        const Material &mtl = scene.materials[hit.materialIndex];
        if(mtl.type == MaterialType::EMISSIVE)
        {
            if(depth == 0 || reflected)
            {
                finalColor += throughput * mtl.color * mtl.intensity;
            }
            else
            {
                float pdf_light = 1.f / scene.nbLights;

                float w = powerHeuristic(lastPdf, pdf_light);

                finalColor += throughput * mtl.color * mtl.intensity * w;
            }
            break;
        }
        BSDFVal bsdf = mtl.getBSDF(ray, hit, rng, isInside, reflected);
        if(bsdf.pdf <= 1e-4f) break;
        if(mtl.type == MaterialType::MIRROR || mtl.type == MaterialType::TRANSPARENT){
            throughput *= bsdf.brdf;
        }
        else{
            int lightIndex = int(curand_uniform(rng) * scene.nbLights);
            lightIndex = min(lightIndex, scene.nbLights - 1);

            const Light& light = scene.lights[lightIndex];
            LightSample ls = light.sample(hit.point, rng, scene);

            if (ls.pdf > 0.f)
            {
                float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
                Ray shadowRay(shadowOrigin, ls.direction);

                if (!scene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
                {

                    float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                    if (cosTheta > 0.f)
                    {
                        float3 f = mtl.evalBSDF(ray, hit, ls.direction);

                        float pdf_light = ls.pdf / scene.nbLights;

                        float pdf_bsdf = mtl.pdf(ray, hit, ls.direction);

                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        finalColor += throughput * f * ls.radiance * cosTheta * w / pdf_light;
                    }
                }
            }

            float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

            throughput = throughput * bsdf.brdf * cosTheta / bsdf.pdf;
            lastPdf = bsdf.pdf;
        }
        if (depth > 3)
        {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = fminf(p, 0.95f);

            if (curand_uniform(rng) > p)
                break;

            throughput /= p;
        }
        float3 origin;
        float sign = reflected ? 1.f : -1.f;
        origin = hit.point + hit.normal * 1e-3f * sign;
        ray = Ray(origin, bsdf.direction);
    }

    return finalColor;
}

__device__ __forceinline__
float3 WhittedIntegrator::toneMap(const float3 &c)
{
    float3 c1 = c * exposure;
    return (c1) / (make_float3(1.f) + c1);
}

__device__ __noinline__
float3 WhittedIntegrator::getSkyColor(const Ray &ray)
{
    float3 rayDir = ray.direction;
    //float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);
    float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);
    float tM = 1 - t;
    float B0 = pow(1.0f - t, 3.0f);
    float B1 = 3.0f * tM * tM * t;
    float B2 = 3.0f * tM * t * t;
    float B3 = t * t * t;

    float mult = B0 * 1.0f + B1 * 1.f + B2 * 1.f + B3 * 20.0f;

    float segmentLength = sizeAtmosphere / skyColorSamples;
    float tCurrent = 0.0f;

    float3 sumR = make_float3(0.f);
    float3 sumM = make_float3(0.f);

    float opticalDepthR = 0.0f;
    float opticalDepthM = 0.0f;

    float mu = dot(rayDir, sunDirection);

    float g = 0.95f;

    float phaseR = (3.0f / (16.0f * GPUPIf)) * (1.0f + mu * mu);

    float temp = 1.0f + g * g - 2.0f * g * mu;
    float phaseM = (3.0f / (8.0f * GPUPIf)) *
                   ((1.0f - g * g) * (1.0f + mu * mu)) /
                   ((2.0f + g * g) * temp * sqrtf(temp));

    for (int i = 0; i < skyColorSamples; ++i)
    {
        float3 samplePosition = ray.origin + rayDir * (tCurrent + segmentLength * 0.5f);

        float height = max(samplePosition.y, 0.0f);

        float hrLocal = expf(-height / hr);
        float hmLocal = expf(-height / hm);

        opticalDepthR += hrLocal * segmentLength;
        opticalDepthM += hmLocal * segmentLength;

        float3 sunSamplePosition = samplePosition;

        float opticalDepthLightR = 0.0f;
        float opticalDepthLightM = 0.0f;

        const int sunSamples = 8;
        float sunSegmentLength = sizeAtmosphere / sunSamples;

        for (int j = 0; j < sunSamples; ++j)
        {
            sunSamplePosition += sunDirection * sunSegmentLength;

            float heightLight = max(sunSamplePosition.y, 0.0f);

            opticalDepthLightR += expf(-heightLight / hr) * sunSegmentLength;
            opticalDepthLightM += expf(-heightLight / hm) * sunSegmentLength;
        }

        float3 tau =
            betaR * (opticalDepthR + opticalDepthLightR) +
            betaM * (opticalDepthM + opticalDepthLightM);

        float3 attenuation =
            make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));

        sumR += attenuation * hrLocal * segmentLength;
        sumM += attenuation * hmLocal * segmentLength;

        tCurrent += segmentLength;
    }

    float3 sky = sumR * betaR * phaseR +
                 sumM * betaM * phaseM * 0.3f;

    float sunAngularRadius = 2.1f * GPUPIf / 180.f;
    float cosTheta = dot(rayDir, sunDirection);

    float sunDisk =
        smoothstep(cos(sunAngularRadius),
                   cos(sunAngularRadius * 0.5f),
                   cosTheta);

    float sunset = pow(1.0f - t, 2.0f);
    float3 sunColor =
        lerp(make_float3(30.f,27.f,24.f),
            make_float3(60.f,25.f,10.f),
            sunset);

    sky += sunColor * sunDisk;

    return sky * mult;
}
