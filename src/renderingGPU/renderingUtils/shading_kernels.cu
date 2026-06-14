#include "shading_kernels.cuh"

__global__ void shadeWavefrontKernel(CudaScene scene, float4 *origins, float4 *directions, float4 *p_throughput,
                                     float4 *p_radiance, int *pixelIndices, RNG *p_rng, bool *p_isInside,
                                     bool *p_lastBounceWasDelta, float *p_lastBsdfPdf, OptixHit *p_hits, int *hitMask,
                                     const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    float4 &throughput = p_throughput[idx];
    float4 &radiance = p_radiance[idx];
    RNG &rng = p_rng[idx];
    bool &isInside = p_isInside[idx];
    bool &lastBounceWasDelta = p_lastBounceWasDelta[idx];
    float &lastBsdfPdf = p_lastBsdfPdf[idx];

    float4 &origin = origins[idx];
    float4 &direction = directions[idx];

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
            radiance += make_float4(0.0f);
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
            float3 samplePosition = make_float3(origin + direction * (tCurrent + segmentLength * 0.5f));

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

        radiance += throughput * make_float4(sky, 0.f);

        return;
    }

    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    OptixHit hit = p_hits[idx];
    Material mtl = scene.materials[hit.materialIndex];

    // ------------------------------------------------------------
    // EMISSIVE
    // ------------------------------------------------------------
    if (mtl.type() == EMISSIVE)
    {
        float3 emission = mtl.color() * mtl.intensity();

        if (lastBounceWasDelta)
        {
            radiance += throughput * make_float4(emission, 0.f);
        }
        else
        {
            float lightPdf = scene.lightPdf(origin, direction);

            float w = powerHeuristic(lastBsdfPdf, lightPdf);

            radiance += throughput * make_float4(emission * w, 0.f);
        }

        return;
    }

    BSDFVal bsdf;

    switch (mtl.type())
    {
    case LAMBERT:
        bsdf = mtl.getLambertBSDF(origin, direction, hit, rng);
        break;

    case METAL:
        bsdf = mtl.getMetalBSDF(origin, direction, hit, rng);
        break;

    case PLASTIC:
        bsdf = mtl.getPlasticBSDF(origin, direction, hit, rng);
        break;

    case MIRROR:
        bsdf = mtl.getMirrorBSDF(origin, direction, hit);
        break;

    case TRANSPARENT:
        bsdf = mtl.getTransparentBSDF(origin, direction, hit, rng, isInside);
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

        float lightSelectionProb = getLightProbability(scene, lightIndex);

        const Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(hit.position, rng, scene);

        if (ls.pdf > 0.f)
        {

            float3 shadowTint = scene.traceShadowRay(hit.position + make_float4(hit.normal, 0.f) * 1e-3f,
                                                     make_float4(ls.direction, 0.f), 1e-3f, ls.distance - 1e-3f);

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
                        pdf_bsdf = mtl.lambertPDF(origin, direction, hit, ls.direction);
                        break;

                    case METAL:
                        f = mtl.evalMetalBSDF(origin, direction, hit, ls.direction);
                        pdf_bsdf = mtl.metalPDF(origin, direction, hit, ls.direction);
                        break;

                    case PLASTIC:
                        f = mtl.evalPlasticBSDF(origin, direction, hit, ls.direction);
                        pdf_bsdf = mtl.plasticPDF(origin, direction, hit, ls.direction);
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

                        radiance += throughput * make_float4(f * ls.radiance * shadowTint * cosTheta * w / pdf_light, 0.f);
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
        throughput *= make_float4(bsdf.brdf, 0.f);
    }
    else
    {
        float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

        throughput *= make_float4(bsdf.brdf * cosTheta / bsdf.pdf,0.f);
    }

    // ------------------------------------------------------------
    // NEXT RAY
    // ------------------------------------------------------------
    origins[idx] = hit.position + bsdf.direction * 1e-3f;
    directions[idx] = bsdf.direction, 0.0f;

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
}

/*__global__
void shadeMissKernel(WavefrontState* states, int* missQueue, int missCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= missCount)
        return;

    int idx = missQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;
    Ray origin, direction = state.origin, direction;
    float3 rayDir = direction;
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
        float3 samplePosition = origin + rayDir * (tCurrent + segmentLength * 0.5f);

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

    radiance += throughput * sky * mult;

    states[idx] = state;
    return;
}

__global__
void shadeEmissiveKernel(CudaScene& gpuScene, WavefrontState* states, OptixHit* hits, int* emissiveQueue, int
emissiveCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= emissiveCount)
        return;

    int idx = emissiveQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    Material mtl = gpuScene.materials[hits[idx].materialIndex];
    float3 emission = mtl.color() * mtl.intensity();

    if(lastBounceWasDelta)
    {
        radiance += throughput * emission;
    }
    else
    {
        float lightPdf = gpuScene.lightPdf(origin, direction);

        float w = powerHeuristic(lastBsdfPdf, lightPdf);

        radiance += throughput * emission * w;
    }

    states[idx] = state;
    return;
}

__global__
void shadeMetalKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    OptixHit* hits,
    int* metalQueue,
    int metalCount,
    int* d_activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= metalCount)
        return;

    int idx = metalQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    OptixHit hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &rng;

    BSDFVal bsdf = mtl.getMetalBSDF(origin, direction, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {

        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng.nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, origin, direction.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalMetalBSDF(origin, direction, hit, ls.direction);

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.metalPDF(origin, direction, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    radiance +=
                        throughput *
                        f *
                        ls.radiance *
                        cosTheta *
                        w /
                        pdf_light;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // BSDF bounce
    // ------------------------------------------------------------
    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput =
        throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.origin, direction = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        origin, direction.time
    );

    lastBounceWasDelta = false;
    lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeLambertKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    OptixHit* hits,
    int* lambertQueue,
    int lambertCount,
    int* d_activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= lambertCount)
        return;

    int idx = lambertQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    OptixHit hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &rng;

    BSDFVal bsdf = mtl.getLambertBSDF(origin, direction, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {

        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng.nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, origin, direction.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalLambertBSDF();

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.lambertPDF(origin, direction, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    radiance +=
                        throughput *
                        f *
                        ls.radiance *
                        cosTheta *
                        w /
                        pdf_light;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // BSDF bounce
    // ------------------------------------------------------------
    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput =
        throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.origin, direction = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        origin, direction.time
    );

    lastBounceWasDelta = false;
    lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadePlasticKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    OptixHit* hits,
    int* plasticQueue,
    int plasticCount,
    int* d_activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= plasticCount)
        return;

    int idx = plasticQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    OptixHit hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &rng;

    BSDFVal bsdf = mtl.getPlasticBSDF(origin, direction, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {

        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng.nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, origin, direction.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalPlasticBSDF(origin, direction, hit, ls.direction);

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.plasticPDF(origin, direction, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    radiance +=
                        throughput *
                        f *
                        ls.radiance *
                        cosTheta *
                        w /
                        pdf_light;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // BSDF bounce
    // ------------------------------------------------------------
    float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

    throughput =
        throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.origin, direction = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        origin, direction.time
    );

    lastBounceWasDelta = false;
    lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeMirrorKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    OptixHit* hits,
    int* mirrorQueue,
    int mirrorCount,
    int* d_activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= mirrorCount)
        return;

    int idx = mirrorQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    OptixHit hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    BSDFVal bsdf = mtl.getMirrorBSDF(origin, direction, hit);

    if (bsdf.pdf <= 1e-4f)
    {

        states[idx] = state;
        return;
    }

    throughput *= bsdf.brdf;

    state.origin, direction = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        origin, direction.time
    );

    lastBounceWasDelta = true;
    lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeTransparentKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    OptixHit* hits,
    int* transparentQueue,
    int transparentCount,
    int* d_activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= transparentCount)
        return;

    int idx = transparentQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray origin, direction = state.origin, direction;
    OptixHit hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &rng;

    BSDFVal bsdf = mtl.getTransparentBSDF(
        origin, direction,
        hit,
        rng,
        isInside
    );

    if (bsdf.pdf <= 1e-4f)
    {

        states[idx] = state;
        return;
    }

    throughput *= bsdf.brdf;

    state.origin, direction = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        origin, direction.time
    );

    lastBounceWasDelta = true;
    lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}
*/