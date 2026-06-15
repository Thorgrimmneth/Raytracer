#include "shading_kernels.cuh"

/*
__global__ void shadeWavefrontKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                     float3 *p_radiance, int *pixelIndices, RNG *p_rng, bool *p_isInside,
                                     bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits, int *hitMask,
                                     const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    float3 &throughput = p_throughput[idx];
    float3 &radiance = p_radiance[idx];
    RNG &rng = p_rng[idx];
    bool &isInside = p_isInside[idx];
    bool &lastBounceWasDelta = p_lastBounceWasDelta[idx];
    float &lastBsdfPdf = p_lastBsdfPdf[idx];

    float3 &origin = origins[idx];
    float3 direction = make_float3(directions[idx]);

    // ------------------------------------------------------------
    // MISS / SKY
    // ------------------------------------------------------------
    if (!hitMask[idx])
    {

        // float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);
        float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);

        const float horizonFade = smoothstep(-0.05f, 0.02f, direction.y);

        if (horizonFade <= 0.0f)
        {
            return;
        }

        float segmentLength = sizeAtmosphere / skyColorSamples;
        float tCurrent = 0.0f;

        float3 sumR = make_float3(0.f);
        float3 sumM = make_float3(0.f);

        float opticalDepthR = 0.0f;
        float opticalDepthM = 0.0f;

        float mu = dot(direction, sunDirection);

        // float g = 0.76f;

        float mu2Term = 1.0f + mu * mu;

        float phaseR = 0.0596831f * mu2Term;
        // float phaseR = (3.0f / (16.0f * GPUPIf)) * (1.0f + mu * mu);

        float temp = 1.5776f - 1.52f * mu;
        // float temp = 1.0f + g * g - 2.0f * g * mu;

        // float phaseM = (3.0f / (8.0f * GPUPIf)) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * temp *
        // sqrtf(temp));
        float phaseM = 0.0195609427f * mu2Term * rsqrtf(temp) / temp;

        const int sunSamples = 4;
        float sunSegmentLength = 15000.f;

        for (int i = 0; i < skyColorSamples; ++i)
        {
            float3 samplePosition = origin + direction * (tCurrent + segmentLength * 0.5f);

            float height = fmaxf(samplePosition.y, 0.0f);

            float hrLocal = __expf(-height * hr);
            float hmLocal = __expf(-height * hm);

            opticalDepthR += hrLocal * segmentLength;
            opticalDepthM += hmLocal * segmentLength;

            float3 sunSamplePosition = samplePosition;

            float opticalDepthLightR = 0.0f;
            float opticalDepthLightM = 0.0f;

            for (int j = 0; j < sunSamples; ++j)
            {
                sunSamplePosition += sunDirection * sunSegmentLength;

                float heightLight = fmaxf(sunSamplePosition.y, 0.0f);

                opticalDepthLightR += __expf(-heightLight * hr) * sunSegmentLength;
                opticalDepthLightM += __expf(-heightLight * hm) * sunSegmentLength;
            }

            float3 mTau =
                -(betaR * (opticalDepthR + opticalDepthLightR) + betaM * (opticalDepthM + opticalDepthLightM));

            float3 attenuation = make_float3(__expf(mTau.x), __expf(mTau.y), __expf(mTau.z));

            sumR += attenuation * hrLocal * segmentLength;
            sumM += attenuation * hmLocal * segmentLength;

            tCurrent += segmentLength;
        }

        float3 sky = sumR * betaR * phaseR + sumM * betaM * phaseM * 0.3f;

        float sunAngularRadius = 2.1f * GPUPIf / 180.f;
        float cosTheta = dot(direction, sunDirection);

        float sunDisk = smoothstep(cos(sunAngularRadius), cos(sunAngularRadius * 0.5f), cosTheta);

        float sunset = (1.f - t) * (1.f - t);
        float3 sunColor = lerp(make_float3(30.f, 27.f, 24.f), make_float3(60.f, 25.f, 10.f), sunset);
        if (!safeSun)
        {
            sunColor = clamp(sunColor, make_float3(0.f), make_float3(1.f));
        }
        sky += sunColor * sunDisk;

        float mult = 1.0f + 19.0f * t * t * t;
        sky = sky * mult * horizonFade;

        radiance += throughput * sky;

        return;
    }

    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    OptixHit &hit = p_hits[idx];
    Material &mtl = scene.materials[hit.materialIndex];

    // ------------------------------------------------------------
    // EMISSIVE
    // ------------------------------------------------------------
    if (mtl.type() == EMISSIVE)
    {
        float3 emission = mtl.color() * mtl.intensity();

        if (lastBounceWasDelta)
        {
            radiance += throughput * emission;
        }
        else
        {
            float lightPdf = scene.lightPdf(origin, direction);

            float w = powerHeuristic(lastBsdfPdf, lightPdf);

            radiance += throughput * emission * w;
        }

        return;
    }

    BSDFVal bsdf;
    float3 &normal = hit.normal;
    switch (mtl.type())
    {
    case LAMBERT:
        bsdf = mtl.getLambertBSDF(direction, normal, rng);
        break;

    case METAL:
        bsdf = mtl.getMetalBSDF(direction, normal, rng);
        break;

    case PLASTIC:
        bsdf = mtl.getPlasticBSDF(direction, normal, rng);
        break;

    case MIRROR:
        bsdf = mtl.getMirrorBSDF(direction, normal);
        break;

    case TRANSPARENT:
        bsdf = mtl.getTransparentBSDF(direction, normal, rng, isInside);
        break;

    default:
        return;
    }

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }

    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------

    if (!bsdf.isDelta && scene.nbLights > 0)
    {
        int lightIndex = selectLightByImportance(scene, rng);

        float lightSelectionProb = getLightProbability(scene.nbLights, scene.lightProbabilities, lightIndex);

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(hit.position, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint =
                scene.traceShadowRay(hit.position + hit.normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f;
                    float pdf_bsdf;

                    switch (mtl.type())
                    {
                    case LAMBERT:
                        f = mtl.evalLambertBSDF();
                        pdf_bsdf = mtl.lambertPDF(direction, normal, ls.direction);
                        break;

                    case METAL:
                        f = mtl.evalMetalBSDF(direction, normal, ls.direction);
                        pdf_bsdf = mtl.metalPDF(direction, normal, ls.direction);
                        break;

                    case PLASTIC:
                        f = mtl.evalPlasticBSDF(direction, normal, ls.direction);
                        pdf_bsdf = mtl.plasticPDF(direction, normal, ls.direction);
                        break;

                    default:
                        f = make_float3(0.f);
                        pdf_bsdf = 0.f;
                        break;
                    }

                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        radiance += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    if (bsdf.isDelta)
    {
        throughput *= bsdf.brdf;
    }
    else
    {
        float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

        throughput *= bsdf.brdf * cosTheta / bsdf.pdf;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origin = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);

    lastBounceWasDelta = bsdf.isDelta;
    lastBsdfPdf = bsdf.pdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}*/

__global__ void shadeMissKernel(float3 *origins, float4 *directions, float3 *throughput, float3 *radiance,
                                const int *missQueue, int missCount, bool safeSun)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= missCount)
        return;

    int idx = missQueue[qid];

    const float3 &origin = origins[idx];
    const float4 &direction = directions[idx];

    const float horizonFade = smoothstep(-0.05f, 0.02f, direction.y);

    if (horizonFade <= 0.0f)
        return;

    const float segmentLength = sizeAtmosphere / skyColorSamples;
    float tCurrent = 0.0f;

    float3 sumR = make_float3(0.f);
    float3 sumM = make_float3(0.f);

    float opticalDepthR = 0.0f;
    float opticalDepthM = 0.0f;

    const float mu = dot4f3(direction, sunDirection);

    const float mu2Term = 1.0f + mu * mu;

    const float phaseR = 0.0596831f * mu2Term;

    const float temp = 1.5776f - 1.52f * mu;

    const float phaseM = 0.0195609427f * mu2Term * rsqrtf(temp) / temp;

    const int sunSamples = 4;
    const float sunSegmentLength = 15000.f;
    const float3 sunDirSegLength = sunDirection * sunSegmentLength;
    for (int i = 0; i < skyColorSamples; ++i)
    {
        float3 samplePosition = origin + direction * (tCurrent + segmentLength * 0.5f);

        float height = fmaxf(samplePosition.y, 0.0f);

        float hrLocal = __expf(-height * hr);
        float hmLocal = __expf(-height * hm);

        opticalDepthR += hrLocal * segmentLength;
        opticalDepthM += hmLocal * segmentLength;

        float3 sunSamplePosition = samplePosition;

        float opticalDepthLightR = 0.f;
        float opticalDepthLightM = 0.f;

        for (int j = 0; j < sunSamples; ++j)
        {
            sunSamplePosition += sunDirSegLength;

            float heightLight = fmaxf(sunSamplePosition.y, 0.f);

            float expR = __expf(-heightLight * hr);
            float expM = __expf(-heightLight * hm);

            opticalDepthLightR = fmaf(expR, sunSegmentLength, opticalDepthLightR);

            opticalDepthLightM = fmaf(expM, sunSegmentLength, opticalDepthLightM);
        }

        float3 tau = -(betaR * (opticalDepthR + opticalDepthLightR) + betaM * (opticalDepthM + opticalDepthLightM));

        float3 attenuation = make_float3(__expf(tau.x), __expf(tau.y), __expf(tau.z));

        sumR += attenuation * hrLocal * segmentLength;
        sumM += attenuation * hmLocal * segmentLength;

        tCurrent += segmentLength;
    }

    float3 sky = sumR * betaR * phaseR + sumM * betaM * phaseM * 0.3f;

    float sunAngularRadius = 2.1f * GPUPIf / 180.f;

    float cosTheta = dot4f3(direction, sunDirection);

    float sunDisk = smoothstep(cos(sunAngularRadius), cos(sunAngularRadius * 0.5f), cosTheta);
    float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);
    float sunset = (1.f - t) * (1.f - t);

    float3 sunColor = lerp(make_float3(30.f, 27.f, 24.f), make_float3(60.f, 25.f, 10.f), sunset);

    if (!safeSun)
    {
        sunColor = clamp(sunColor, make_float3(0.f), make_float3(1.f));
    }

    sky += sunColor * sunDisk;

    float mult = 1.0f + 19.0f * t * t * t;

    sky = sky * mult * horizonFade;

    float3 &T = throughput[idx];
    float3 &L = radiance[idx];

    L += T * sky;
}

__global__ void shadeLambertKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                   float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                   int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 &throughput = p_throughput[idx];
    float3 &radiance = p_radiance[idx];

    float3 &origin = origins[idx];

    float3 direction = make_float3(directions[idx]);

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    RNG &rng = p_rng[idx];
    float3 &normal = hit.normal;
    BSDFVal bsdf = mtl.getLambertBSDF(direction, normal, rng);

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }
    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------

    if (scene.nbLights > 0)
    {
        int lightIndex = selectLightByImportance(scene, rng);

        float lightSelectionProb = getLightProbability(scene.nbLights, scene.lightProbabilities, lightIndex);

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(hit.position, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint =
                scene.traceShadowRay(hit.position + hit.normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalLambertBSDF();
                    float pdf_bsdf = mtl.lambertPDF(direction, normal, ls.direction);

                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        radiance += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput *= bsdf.brdf * cosTheta / bsdf.pdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origin = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);
    lastBounceWasDelta[idx] = false;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMetalKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                 float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 &throughput = p_throughput[idx];
    float3 &radiance = p_radiance[idx];

    float3 &origin = origins[idx];

    float3 direction = make_float3(directions[idx]);

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    RNG &rng = p_rng[idx];
    float3 &normal = hit.normal;
    BSDFVal bsdf = mtl.getMetalBSDF(direction, normal, rng);

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }

    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------

    if (scene.nbLights > 0)
    {
        int lightIndex = selectLightByImportance(scene, rng);

        float lightSelectionProb = getLightProbability(scene.nbLights, scene.lightProbabilities, lightIndex);

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(hit.position, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint =
                scene.traceShadowRay(hit.position + hit.normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalMetalBSDF(direction, normal, ls.direction);
                    float pdf_bsdf = mtl.metalPDF(direction, normal, ls.direction);

                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        radiance += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput *= bsdf.brdf * cosTheta / bsdf.pdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origin = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);
lastBounceWasDelta[idx] = false;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadePlasticNEEKernel(CudaScene &scene, float4 *directions, float3 *p_throughput, float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, const int *hitQueue, int hitCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 direction = make_float3(directions[idx]);

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    RNG &rng = p_rng[idx];
    float3 &normal = hit.normal;

    if (scene.nbLights > 0)
    {
        int lightIndex = selectLightByImportance(scene, rng);

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(hit.position, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint =
                scene.traceShadowRay(hit.position + hit.normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalPlasticBSDF(direction, normal, ls.direction);
                    float pdf_bsdf = mtl.plasticPDF(direction, normal, ls.direction);
                    float lightSelectionProb = getLightProbability(scene.nbLights, scene.lightProbabilities, lightIndex);
                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        p_radiance[idx] += p_throughput[idx] * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

}
__global__ void shadePlasticKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                    RNG *p_rng, OptixHit *p_hits, bool *p_lastBounceWasDelta, const int *hitQueue, int hitCount,
                                   int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 &throughput = p_throughput[idx];

    float3 direction = make_float3(directions[idx]);

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    RNG &rng = p_rng[idx];
    float3 &normal = hit.normal;
    BSDFVal bsdf = mtl.getPlasticBSDF(direction, normal, rng); //beaucoup de registres

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }
    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput *= bsdf.brdf * cosTheta / bsdf.pdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origins[idx] = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);
    p_lastBounceWasDelta[idx] = false;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMirrorKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput, RNG *p_rng,
                                  bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits,
                                  const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                  uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 &throughput = p_throughput[idx];

    float3 direction = make_float3(directions[idx]);

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    float3 &normal = hit.normal;
    BSDFVal bsdf = mtl.getMirrorBSDF(direction, normal);

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    throughput *= bsdf.brdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        RNG &rng = p_rng[idx];
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origins[idx] = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);

    p_lastBounceWasDelta[idx] = true;
    p_lastBsdfPdf[idx] = bsdf.pdf;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeTransparentKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                       RNG *p_rng, bool *p_isInside,
                                       bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits,
                                       const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                       uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 &throughput = p_throughput[idx];

    bool &isInside = p_isInside[idx];

    OptixHit &hit = p_hits[idx];

    Material &mtl = scene.materials[hit.materialIndex];

    BSDFVal bsdf = mtl.getTransparentBSDF(make_float3(directions[idx]), hit.normal, p_rng[idx], isInside);

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    throughput *= bsdf.brdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        RNG &rng = p_rng[idx];
        if (rng.nextFloat() > p)
        {
            return;
        }

        throughput /= p;
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    float3 dir = bsdf.direction;
    origins[idx] = hit.position + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);

    p_lastBounceWasDelta[idx] = true;
    p_lastBsdfPdf[idx] = bsdf.pdf;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeEmissiveKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                    float3 *p_radiance, bool *p_lastBounceWasDelta,
                                    float *p_lastBsdfPdf, OptixHit *p_hits, int *hitMask, const int *activeQueue,
                                    int activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    float3 &throughput = p_throughput[idx];
    float3 &radiance = p_radiance[idx];
    
    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    OptixHit &hit = p_hits[idx];
    Material &mtl = scene.materials[hit.materialIndex];

    // ------------------------------------------------------------
    // EMISSIVE
    // ------------------------------------------------------------

    float3 emission = mtl.color() * mtl.intensity();

    if (p_lastBounceWasDelta[idx])
    {
        radiance += throughput * emission;
    }
    else
    {
        float lightPdf = scene.lightPdf(origins[idx], make_float3(directions[idx]));

        float w = powerHeuristic(p_lastBsdfPdf[idx], lightPdf);

        radiance += throughput * emission * w;
    }

    return;
}