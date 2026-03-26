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

    for (int depth = 0; depth < nbBounces; depth++)
    {
        HitRecord hit;

        if (!scene.intersect(ray, tMin, tMax, hit))
        {
            finalColor += throughput * getSkyColor(ray);
            break;
        }

        const Material &mtl = scene.materials[hit.materialIndex];
        if(mtl.type == MaterialType::EMISSIVE){
            finalColor += throughput * mtl.color * mtl.intensity;
            break;
        }
        BSDFVal bsdf = mtl.getBSDF(ray, hit, rng);

        if(mtl.type == MaterialType::MIRROR || mtl.type == MaterialType::TRANSPARENT){
            throughput *= bsdf.brdf;
        }
        else{
            float cosTheta = fabs(dot(hit.normal, bsdf.direction));

            throughput = throughput * bsdf.brdf * cosTheta / bsdf.pdf;
        }

        if (dot(bsdf.direction, hit.normal) < 0.f)
            ray.origin = hit.point - hit.normal * 1e-3f;
        else
            ray.origin = hit.point + hit.normal * 1e-3f;
        ray.direction = bsdf.direction;
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
    float3 rayDir = normalize(ray.direction);
    float3 sunDir = normalize(sunDirection);
    //float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);
    float t = clamp((sunDir.y + 0.4f) / 1.4f, 0.0f, 1.0f);
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

    float mu = dot(rayDir, sunDir);

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
            sunSamplePosition += sunDir * sunSegmentLength;

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
    float cosTheta = dot(rayDir, sunDir);

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
