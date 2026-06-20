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

__global__ void shadeMissKernel(float4 *origins, float4 *directions, float4 *throughput, float3 *radiance,
                                const int *missQueue, int missCount, bool safeSun)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= missCount)
        return;

    int idx = missQueue[qid];

    const float3 origin = make_float3(origins[idx]);
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

    float4 thru4 = throughput[idx];
    float3 T = make_float3(thru4);
    float3 &L = radiance[idx];

    L += T * sky;
}

__global__ void shadeLambertKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                   float3 *p_radiance, RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                   int *p_hitMaterialIndices, bool *lastBounceWasDelta, const int *hitQueue,
                                   int hitCount, int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];
    float3 direction = make_float3(directions[idx]);
    float4 thru4 = p_throughput[idx];
    float3 throughput = make_float3(thru4);
    int materialIndex = p_hitMaterialIndices[idx];
    float3 pos = make_float3(p_hitPositions[idx]);
    float3 normal = make_float3(p_hitNormals[idx]);

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

            float3 shadowTint =
                scene.traceShadowRay(pos + normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f); // problèmes

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
    origins[idx] = make_float4(pos + dir * 1e-3f, 0.f);
    directions[idx] = make_float4(dir, 0.f);
    p_throughput[idx] = make_float4(throughput, thru4.w);

    lastBounceWasDelta[idx] = false;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMetalKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                 float3 *p_radiance, RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                 int *p_hitMaterialIndices, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float4 thru4 = p_throughput[idx];
    float3 throughput = make_float3(thru4);
    float3 &radiance = p_radiance[idx];

    float4 origin4 = origins[idx];
    float3 origin = make_float3(origin4);

    float3 direction = make_float3(directions[idx]);

    int materialIndex = p_hitMaterialIndices[idx];
    float3 pos = make_float3(p_hitPositions[idx]);

    Material &mtl = scene.materials[materialIndex];

    RNG &rng = p_rng[idx];
    float3 normal = make_float3(p_hitNormals[idx]);
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

        LightSample ls = light.sample(pos, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint = scene.traceShadowRay(pos + normal * 1e-3f, ls.direction, 1e-3f, ls.distance - 1e-3f);

            if (length(shadowTint) > 1e-6f)
            {
                float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);

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
    origins[idx] = make_float4(pos + dir * 1e-3f, 0.f);
    directions[idx] = make_float4(dir, 0.f);
    p_throughput[idx] = make_float4(throughput, thru4.w);
    lastBounceWasDelta[idx] = false;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadePlasticNEEKernel(CudaScene &scene, float4 *directions, float4 *p_throughput, float3 *p_radiance,
                                      RNG *p_rng, float4 *p_hitPositions, float4 *p_hitNormals,
                                      int *p_hitMaterialIndices, const int *hitQueue, int hitCount, int nbLights)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float4 dir4 = directions[idx];
    float4 thru4 = p_throughput[idx];
    float3 throughput = make_float3(thru4);

    float3 pos = make_float3(p_hitPositions[idx]);
    int materialIndex = p_hitMaterialIndices[idx];
    float3 normal = make_float3(p_hitNormals[idx]);
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
                        p_radiance[idx] += throughput * f * ls.radiance * shadowTint * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }
}
__global__ void shadePlasticKernel(Material *materials,
                                   float4 *origins, // float4 pour coalescence
                                   float4 *directions,
                                   float4 *throughput, // float4 au lieu de float3
                                   RNG *rng, float4 *hitPositions,
                                   float4 *hitNormals, // XYZ = normal, W = padding
                                   int *hitMaterialIndices, bool *lastBounceWasDelta, const int *hitQueue, int hitCount,
                                   int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    // Load une seule fois (évite les re-lectures)
    float4 pos4 = hitPositions[idx];
    float4 normal4 = hitNormals[idx];
    float4 dir4 = directions[idx];
    float4 thru4 = throughput[idx];
    int matIdx = hitMaterialIndices[idx];
    RNG rng_local = rng[idx];

    // Extraire composantes (le compilateur optimise ça en registres)
    float3 pos = make_float3(pos4);
    float3 normal = make_float3(normal4);
    float3 dir = make_float3(dir4);
    float3 thru = make_float3(thru4);

    // Material inline (pas de struct en mémoire)
    float alpha = materials[matIdx].alpha();
    float alphaSquared = alpha * alpha;
    float3 baseColor = materials[matIdx].color();

    float3 wo = -dir;

    // === FRESNEL (inliner complètement) ===
    float cosThetaView = saturate(dot(normal, wo));
    float cosT5 = pow5(1.f - cosThetaView); // Utiliser une fonction pow5 optimisée
    float specW = 0.04f + 0.96f * cosT5;    // Fresnel scalar directement
    float3 F = make_float3(specW);

    // === SAMPLING ===
    float e1 = rng_local.nextFloat();
    float e2 = rng_local.nextFloat();

    // Tangent frame (optimisé avec branchement minimal)
    float3 T, B;
    getTangentFrame(normal, T, B); // Fonction inline

    float3 newDir;
    float3 brdf;
    float pdf;

    if (rng_local.nextFloat() < specW)
    {
        // SPECULAR (GGX)
        // Éviter d'allouer V en registre - calculer directement
        float Vx = dot(wo, T);
        float Vy = dot(wo, B);
        float Vz = dot(wo, normal);

        // Normaliser inline sans struct
        float lenV = rsqrtf(Vx * Vx + Vy * Vy + Vz * Vz + 1e-8f);
        Vx *= lenV;
        Vy *= lenV;
        Vz *= lenV;

        // Stretch
        Vx *= alpha;
        Vy *= alpha; // Vz reste

        // Sample GGX
        float3 h = sampleGGX(e1, e2, Vx, Vy, Vz); // Retourne normalité

        // Transform back
        float3 H = h.x * T + h.y * B + h.z * normal;

        newDir = reflect(-wo, H);

        if (dot(normal, newDir) <= 0.f)
        {
            pdf = 0.f;
            brdf = make_float3(0.f);
        }
        else
        {
            // Évaluer GGX sans redondance
            float NdotH = dot(normal, H);
            float NdotV = dot(normal, wo);
            float NdotL = dot(normal, newDir);
            float HdotV = dot(H, wo);

            // Calculer D, G, F en une passe
            float denom = fmaxf(NdotH * NdotH * (alphaSquared - 1.f) + 1.f, 1e-8f);
            float D = alphaSquared / (GPUPIf * denom * denom);

            // Smith G (sans réallouer V)
            float G1V = 2.f / (1.f + sqrtf(1.f + alphaSquared * (1.f - NdotV * NdotV) / fmaxf(NdotV * NdotV, 1e-8f)));
            float G1L = 2.f / (1.f + sqrtf(1.f + alphaSquared * (1.f - NdotL * NdotL) / fmaxf(NdotL * NdotL, 1e-8f)));
            float G = G1V * G1L;

            // Fresnel pour specular
            float cosT5_h = pow5(1.f - HdotV);
            float3 Fspec = make_float3(0.04f + 0.96f * cosT5_h);

            brdf = (D * G / fmaxf(4.f * NdotV * NdotL, 1e-8f)) * Fspec;
            pdf = (dot(wo, H) > 0.f && NdotV > 0.f) ? G1V * D / fmaxf(4.f * NdotV, 1e-8f) : 0.f;
            pdf = specW * pdf + (1.f - specW) * fmaxf(NdotL, 0.f) * GPUInvPIf;
        }
    }
    else
    {
        // DIFFUSE (cosine sampling)
        float r = sqrtf(e1);
        float phi = 2.f * GPUPIf * e2;
        float sint = r;
        float cost = sqrtf(1.f - sint * sint);
        newDir = normalize(sint * cosf(phi) * T + sint * sinf(phi) * B + cost * normal);
        brdf = baseColor * GPUInvPIf;
        pdf = (1.f - specW) * fmaxf(dot(normal, newDir), 0.f) * GPUInvPIf;
    }

    // === EARLY EXIT ===
    if (pdf <= 1e-4f)
        return;

    // === UPDATE THROUGHPUT ===
    float cosTheta = fmaxf(dot(normal, newDir), 0.f);
    thru *= brdf * cosTheta / pdf;

    // === RUSSIAN ROULETTE ===
    if (depth > 2)
    {
        float p = fmaxf(thru.x, fmaxf(thru.y, thru.z));
        p = clamp(p, 0.1f, 1.f);
        if (rng_local.nextFloat() > p)
            return;
        thru /= p;
    }

    // === WRITE BACK (une seule fois, coalescé) ===
    origins[idx] = make_float4(pos + newDir * 1e-3f, 0.f);
    directions[idx] = make_float4(newDir, 0.f);
    throughput[idx] = make_float4(thru, thru4.w); // Préserver le .w
    lastBounceWasDelta[idx] = false;

    // === QUEUE DISPATCH ===
    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeMirrorKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput, RNG *p_rng,
                                 bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, float4 *p_hitPositions,
                                 float4 *p_hitNormals, int *p_hitMaterialIndices, const int *hitQueue, int hitCount,
                                 int *nextActiveQueue, int *nextActiveCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float3 throughput = make_float3(p_throughput[idx]);

    float3 direction = make_float3(directions[idx]);

    int materialIndex = p_hitMaterialIndices[idx];
    float3 pos = make_float3(p_hitPositions[idx]);

    Material &mtl = scene.materials[materialIndex];

    float3 normal = make_float3(p_hitNormals[idx]);
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
    origins[idx] = make_float4(pos + dir * 1e-3f, 0.f);
    directions[idx] = make_float4(dir, 0.f);

    p_lastBounceWasDelta[idx] = true;
    p_lastBsdfPdf[idx] = bsdf.pdf;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeTransparentKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                       RNG *p_rng, bool *p_isInside, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                       float4 *p_hitPositions, float4 *p_hitNormals, int *p_hitMaterialIndices,
                                       const int *hitQueue, int hitCount, int *nextActiveQueue, int *nextActiveCount,
                                       uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= hitCount)
        return;

    int idx = hitQueue[qid];

    float4 thru4 = p_throughput[idx];
    float3 throughput = make_float3(thru4);

    bool &isInside = p_isInside[idx];

    int materialIndex = p_hitMaterialIndices[idx];
    float3 pos = make_float3(p_hitPositions[idx]);
    float3 normal = make_float3(p_hitNormals[idx]);

    Material &mtl = scene.materials[materialIndex];

    BSDFVal bsdf = mtl.getTransparentBSDF(make_float3(directions[idx]), normal, p_rng[idx], isInside);

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
    origins[idx] = make_float4(pos + dir * 1e-3f, 0.f);
    directions[idx] = make_float4(dir, 0.f);
    p_throughput[idx] = make_float4(throughput, thru4.w);

    p_lastBounceWasDelta[idx] = true;
    p_lastBsdfPdf[idx] = bsdf.pdf;

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;
}

__global__ void shadeEmissiveKernel(CudaScene &scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                    float3 *p_radiance, bool *p_lastBounceWasDelta, float *p_lastBsdfPdf,
                                    int *p_hitMaterialIndices, int *hitMask, const int *activeQueue, int activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    float4 thru4 = p_throughput[idx];
    float3 throughput = make_float3(thru4);
    float3 &radiance = p_radiance[idx];

    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    int materialIndex = p_hitMaterialIndices[idx];
    Material &mtl = scene.materials[materialIndex];

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
        float lightPdf = scene.lightPdf(make_float3(origins[idx]), make_float3(directions[idx]));

        float w = powerHeuristic(p_lastBsdfPdf[idx], lightPdf);

        radiance += throughput * emission * w;
    }

    return;
}