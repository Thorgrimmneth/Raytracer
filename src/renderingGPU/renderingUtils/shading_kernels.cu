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
                                   float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta,
                                   const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                   uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];
    float3 direction = make_float3(directions[idx]);
    float3 &throughput = p_throughput[idx];
    int materialIndex = p_hits[idx].materialIndex;
    float3 pos = p_hits[idx].position;
    float3 normal = p_hits[idx].normal;

    Material &mtl = scene.materials[materialIndex];

    RNG &rng = p_rng[idx];
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

        LightSample ls = light.sample(pos, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint = scene.traceShadowRay(pos + normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalLambertBSDF();
                    float pdf_bsdf = mtl.lambertPDF(direction, normal, ls.direction);

                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);
                        float3 &radiance = p_radiance[idx];
                        radiance += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

    float cosTheta = fmaxf(dot(normal, bsdf.direction), 0.0f);
    
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
    origins[idx] = pos + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);

    lastBounceWasDelta[idx] = false;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMetalKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                 float3 *p_radiance, RNG *p_rng, OptixHit *p_hits, bool *lastBounceWasDelta,
                                 const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                 uint depth)
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

__global__ void shadePlasticNEEKernel(CudaScene &scene, float4 *directions, float3 *p_throughput, float3 *p_radiance,
                                      RNG *p_rng, OptixHit *p_hits, const int *hitQueue, int hitCount, int nbLights)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float4 dir4 = directions[idx];

    float3 pos = p_hits[idx].position;
    int materialIndex = p_hits[idx].materialIndex;
    float3 normal = p_hits[idx].normal;
    Material &mtl = scene.materials[materialIndex];

    RNG &rng = p_rng[idx];

    if (nbLights > 0)
    {
        int lightIndex = 0;

        // Get total weight from last entry
        float totalWeight = scene.lightCumulativeWeights[scene.nbLights - 1];

        if (totalWeight <= 0.0f)
        {
            // Fallback to uniform selection if no lights have intensity
            lightIndex = min(int(rng.nextFloat() * scene.nbLights), nbLights - 1);
        }
        else
        {
            // Binary search in cumulative weights array
            float random = rng.nextFloat() * totalWeight;

            int left = 0;
            int right = nbLights - 1;

            while (left < right)
            {
                int mid = (left + right) / 2;
                if (scene.lightCumulativeWeights[mid] < random)
                    left = mid + 1;
                else
                    right = mid;
            }
            lightIndex = left;
        }

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(pos, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint = scene.traceShadowRay(pos + normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 direction = make_float3(dir4);
                    float3 f = mtl.evalPlasticBSDF(direction, normal, ls.direction);
                    float pdf_bsdf = mtl.plasticPDF(direction, normal, ls.direction);
                    float lightSelectionProb = (lightIndex >= nbLights) ? 1.f : scene.lightProbabilities[lightIndex];
                    float pdf_light = ls.pdf * lightSelectionProb;

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);
                        float3 throughput = p_throughput[idx];
                        p_radiance[idx] += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }
}
__global__ void shadePlasticKernel(CudaScene &scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                   RNG *p_rng, OptixHit *p_hits, bool *p_lastBounceWasDelta, const int *hitQueue,
                                   int hitCount, int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float4 dir4 = directions[idx];
    float3 normal = p_hits[idx].normal;
    int materialIndex = p_hits[idx].materialIndex;
    float3 pos = p_hits[idx].position;
    RNG rng = p_rng[idx];

    // ne pas toucher aux données chargées tout de suite
    float3 &throughput = p_throughput[idx];

    // extraire ce qui servira plus tard

    Material mtl = scene.materials[materialIndex];

    // charger les paramètres matériau immédiatement
    float alpha = mtl.alpha();
    float alphaSquared = alpha * alpha;
    float3 baseColor = mtl.color();

    // seulement maintenant utiliser les données chargées

    float3 wo = -make_float3(dir4);
    // ============================================================
    // getPlasticBSDF INLINE
    // ============================================================

    BSDFVal bsdf;

    float cosThetaView = saturate(dot(normal, wo));

    float3 F0 = make_float3(0.04f);
    float t = 1.f - cosThetaView;
    float t2 = t * t;
    float t4 = t2 * t2;
    float t5 = t4 * t;

    float3 F = F0 + (make_float3(0.96f)) * t5;

    float specW = (F.x + F.y + F.z) * (1.f / 3.f);

    float e1 = rng.nextFloat();
    float e2 = rng.nextFloat();
    float sign = copysignf(1.0f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    float3 T = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 B = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
    if (rng.nextFloat() < specW)
    {
        // ========================================================
        // samplingGGX INLINE
        // ========================================================

        float alphaC = fmaxf(mtl.alpha(), 1e-4f);

        float3 V = normalize(make_float3(dot(wo, T), dot(wo, B), dot(wo, normal)));

        V = normalize(make_float3(alphaC * V.x, alphaC * V.y, V.z));

        float3 T1 = (V.z < 0.9999f) ? normalize(cross(make_float3(0.f, 0.f, 1.f), V)) : make_float3(1.f, 0.f, 0.f);

        float3 T2 = cross(V, T1);

        float r = sqrtf(e1);
        float phi = 2.f * GPUPIf * e2;

        float t1 = r * cosf(phi);
        float t2 = r * sinf(phi);

        float s = 0.5f * (1.f + V.z);

        t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

        float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * V;

        float3 h = normalize(make_float3(alphaC * Nh.x, alphaC * Nh.y, fmaxf(0.f, Nh.z)));

        h = h.x * T + h.y * B + h.z * normal;

        bsdf.direction = reflect(-wo, h);

        if (dot(normal, bsdf.direction) <= 0.f)
        {
            bsdf.pdf = 0.f;
            bsdf.brdf = make_float3(0.f);
        }
        else
        {
            // ====================================================
            // evaluateGGX INLINE
            // ====================================================

            float3 hEval = normalize(bsdf.direction + wo);

            float NdotV = fmaxf(dot(normal, wo), 0.f);

            float NdotL = fmaxf(dot(normal, bsdf.direction), 0.f);

            if (NdotV <= 0.f || NdotL <= 0.f)
            {
                bsdf.brdf = make_float3(0.f);
            }
            else
            {
                float NdotH = fmaxf(dot(normal, hEval), 0.f);

                float NdotH2 = NdotH * NdotH;

                float denomD = (NdotH2 * (alphaSquared - 1.f) + 1.f);

                float D = alphaSquared / (GPUPIf * denomD * denomD);

                float HdotV = clamp(dot(hEval, wo), 0.f, 1.f);

                float ft = 1.f - HdotV;
                float ft2 = ft * ft;
                float ft4 = ft2 * ft2;
                float ft5 = ft4 * ft;

                float3 Fspec = F0 + make_float3(0.96f) * ft5;

                // =================================================
                // computeG INLINE
                // =================================================

                float G1V;
                {
                    float NdotV2 = NdotV * NdotV;
                    float tan2 = (1.f - NdotV2) / fmaxf(NdotV2, 1e-8f);

                    G1V = 2.f / (1.f + sqrtf(1.f + alphaSquared * tan2));
                }

                float G1L;
                {
                    float NdotL2 = NdotL * NdotL;
                    float tan2 = (1.f - NdotL2) / fmaxf(NdotL2, 1e-8f);

                    G1L = 2.f / (1.f + sqrtf(1.f + alphaSquared * tan2));
                }

                float G = G1V * G1L;

                float denom = 4.f * NdotV * NdotL;

                bsdf.brdf = (D * G / denom) * Fspec;
            }

            // ====================================================
            // pdfGGX INLINE
            // ====================================================

            float pdf = 0.f;

            if (dot(wo, hEval) > 0.f)
            {

                if (NdotV > 0.f)
                {
                    float NdotH = fmaxf(dot(normal, hEval), 0.f);

                    float NdotH2 = NdotH * NdotH;

                    float denomD = (NdotH2 * (alphaSquared - 1.f) + 1.f);

                    float D = alphaSquared / (GPUPIf * denomD * denomD);

                    float NdotV2 = NdotV * NdotV;

                    float tan2 = (1.f - NdotV2) / fmaxf(NdotV2, 1e-8f);

                    float G1 = 2.f / (1.f + sqrtf(1.f + alphaSquared * tan2));

                    pdf = (G1 * D) / (4.f * NdotV);
                }
            }

            float diffusePdf = fmaxf(dot(normal, bsdf.direction), 0.f) * GPUInvPIf;

            bsdf.pdf = specW * pdf + (1.f - specW) * diffusePdf;
        }
    }
    else
    {
        float r = sqrtf(e1);
        float phi = 2.f * GPUPIf * e2;

        float x = r * cosf(phi);
        float y = r * sinf(phi);
        float z = sqrtf(fmaxf(0.f, 1.f - x * x - y * y));

        bsdf.direction = normalize(x * T + y * B + z * normal);

        bsdf.brdf = baseColor * GPUInvPIf;

        float diffusePdf = fmaxf(dot(normal, bsdf.direction), 0.f) * GPUInvPIf;

        bsdf.pdf = (1.f - specW) * diffusePdf;
    }

    if (bsdf.pdf <= 1e-4f)
    {
        return;
    }
    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    float cosTheta = fmaxf(dot(normal, bsdf.direction), 0.0f);

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
    origins[idx] = pos + dir * 1e-3f;
    directions[idx] = make_float4(dir, 0.f);
    p_lastBounceWasDelta[idx] = false;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMirrorKernel(CudaScene scene, float3 *origins, float4 *directions, float3 *p_throughput,
                                  RNG *p_rng, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits,
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
                                       RNG *p_rng, bool *p_isInside, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                       OptixHit *p_hits, const int *hitQueue, int hitCount, int *nextActiveQueue,
                                       int *nextActiveCount, uint depth)
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
                                    float3 *p_radiance, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                    OptixHit *p_hits, int *hitMask, const int *activeQueue, int activeCount)
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