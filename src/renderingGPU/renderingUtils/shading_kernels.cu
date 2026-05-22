#include "shading_kernels.cuh"

__global__ void shadeWavefrontKernel(CudaScene scene, WavefrontState *states, HitRecord *hits, int *hitMask,
                                     const int *activeQueue, int activeCount, int *nextActiveQueue,
                                     int *nextActiveCount, bool safeSun)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray ray = state.ray;

    // ------------------------------------------------------------
    // MISS / SKY
    // ------------------------------------------------------------
    if (!hitMask[idx])
    {
        float3 rayDir = ray.direction;

        // float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);
        float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);

        float mult = 1.0f + 19.0f * t * t * t;

        const float horizonFade = smoothstep(-0.05f, 0.02f, rayDir.y);

        if (horizonFade <= 0.0f)
        {
            state.radiance += state.throughput * make_float3(0.0f);
            state.terminate();

            states[idx] = state;
            return;
        }

        float segmentLength = sizeAtmosphere / skyColorSamples;
        float tCurrent = 0.0f;

        float3 sumR = make_float3(0.f);
        float3 sumM = make_float3(0.f);

        float opticalDepthR = 0.0f;
        float opticalDepthM = 0.0f;

        float mu = dot(rayDir, sunDirection);

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
            float3 samplePosition = ray.origin + rayDir * (tCurrent + segmentLength * 0.5f);

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
        float cosTheta = dot(rayDir, sunDirection);

        float sunDisk = smoothstep(cos(sunAngularRadius), cos(sunAngularRadius * 0.5f), cosTheta);

        float sunset = pow(1.0f - t, 2.0f);
        float3 sunColor = lerp(make_float3(30.f, 27.f, 24.f), make_float3(60.f, 25.f, 10.f), sunset);
        if (!safeSun)
        {
            sunColor = clamp(sunColor, make_float3(0.f), make_float3(1.f));
        }
        sky += sunColor * sunDisk;

        sky *mult *horizonFade;

        state.radiance += state.throughput * sky * mult;
        state.terminate();

        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    HitRecord hit = hits[idx];
    Material mtl = scene.materials[hit.materialIndex];

    // ------------------------------------------------------------
    // EMISSIVE
    // ------------------------------------------------------------
    if (mtl.type() == EMISSIVE)
    {
        float3 emission = mtl.color() * mtl.intensity();

        if (state.lastBounceWasDelta)
        {
            state.radiance += state.throughput * emission;
        }
        else
        {
            float lightPdf = scene.lightPdf(ray.origin, ray.direction);

            float w = powerHeuristic(state.lastBsdfPdf, lightPdf);

            state.radiance += state.throughput * emission * w;
        }

        state.terminate();
        states[idx] = state;
        return;
    }

    RNG *rng = &state.rng;

    BSDFVal bsdf;

    switch (mtl.type())
    {
    case LAMBERT:
        bsdf = mtl.getLambertBSDF(ray, hit, rng);
        break;

    case METAL:
        bsdf = mtl.getMetalBSDF(ray, hit, rng);
        break;

    case PLASTIC:
        bsdf = mtl.getPlasticBSDF(ray, hit, rng);
        break;

    case MIRROR:
        bsdf = mtl.getMirrorBSDF(ray, hit);
        break;

    case TRANSPARENT:
        bsdf = mtl.getTransparentBSDF(ray, hit, rng, state.isInside);
        break;

    default:
        state.terminate();
        states[idx] = state;
        return;
    }

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------

    if (!bsdf.isDelta && scene.nbLights > 0)
    {
        int lightIndex = int(rng->nextFloat() * scene.nbLights);
        lightIndex = min(lightIndex, scene.nbLights - 1);

        const Light &light = scene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, scene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;

            Ray shadowRay(shadowOrigin, ls.direction, ray.time);

            if (!scene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
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
                        pdf_bsdf = mtl.lambertPDF(ray, hit, ls.direction);
                        break;

                    case METAL:
                        f = mtl.evalMetalBSDF(ray, hit, ls.direction);
                        pdf_bsdf = mtl.metalPDF(ray, hit, ls.direction);
                        break;

                    case PLASTIC:
                        f = mtl.evalPlasticBSDF(ray, hit, ls.direction);
                        pdf_bsdf = mtl.plasticPDF(ray, hit, ls.direction);
                        break;

                    default:
                        f = make_float3(0.f);
                        pdf_bsdf = 0.f;
                        break;
                    }

                    float pdf_light = ls.pdf * (1.f / scene.nbLights);

                    if (pdf_light > 0.f)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        state.radiance += state.throughput * f * ls.radiance * cosTheta * w / pdf_light;
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
        state.throughput *= bsdf.brdf;
    }
    else
    {
        float cosTheta = fmaxf(dot(hit.normal, bsdf.direction), 0.0f);

        state.throughput = state.throughput * bsdf.brdf * cosTheta / bsdf.pdf;
    }

    // ------------------------------------------------------------
    // NEXT RAY
    // ------------------------------------------------------------

    state.ray = Ray(hit.point + bsdf.direction * 1e-3f, bsdf.direction, ray.time);

    state.depth += 1;

    state.lastBounceWasDelta = bsdf.isDelta;
    state.lastBsdfPdf = bsdf.pdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (state.depth > 3)
    {
        float p = fmaxf(state.throughput.x, fmaxf(state.throughput.y, state.throughput.z));

        p = clamp(p, 0.05f, 0.95f);

        if (rng->nextFloat() > p)
        {
            state.terminate();
            states[idx] = state;
            return;
        }

        state.throughput /= p;
    }

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------

    int dst = atomicAdd(nextActiveCount, 1);
    nextActiveQueue[dst] = idx;

    states[idx] = state;
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
    Ray ray = state.ray;
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

    state.radiance += state.throughput * sky * mult;
    state.terminate();
    states[idx] = state;
    return;
}

__global__
void shadeEmissiveKernel(CudaScene& gpuScene, WavefrontState* states, HitRecord* hits, int* emissiveQueue, int
emissiveCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= emissiveCount)
        return;

    int idx = emissiveQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
        return;

    Ray ray = state.ray;
    Material mtl = gpuScene.materials[hits[idx].materialIndex];
    float3 emission = mtl.color() * mtl.intensity();

    if(state.lastBounceWasDelta)
    {
        state.radiance += state.throughput * emission;
    }
    else
    {
        float lightPdf = gpuScene.lightPdf(ray.origin, ray.direction);

        float w = powerHeuristic(state.lastBsdfPdf, lightPdf);

        state.radiance += state.throughput * emission * w;
    }
    state.terminate();
    states[idx] = state;
    return;
}

__global__
void shadeMetalKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    HitRecord* hits,
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

    Ray ray = state.ray;
    HitRecord hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &state.rng;

    BSDFVal bsdf = mtl.getMetalBSDF(ray, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng->nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, ray.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalMetalBSDF(ray, hit, ls.direction);

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.metalPDF(ray, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    state.radiance +=
                        state.throughput *
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

    state.throughput =
        state.throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.ray = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        ray.time
    );

    state.lastBounceWasDelta = false;
    state.lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeLambertKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    HitRecord* hits,
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

    Ray ray = state.ray;
    HitRecord hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &state.rng;

    BSDFVal bsdf = mtl.getLambertBSDF(ray, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng->nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, ray.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalLambertBSDF();

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.lambertPDF(ray, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    state.radiance +=
                        state.throughput *
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

    state.throughput =
        state.throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.ray = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        ray.time
    );

    state.lastBounceWasDelta = false;
    state.lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadePlasticKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    HitRecord* hits,
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

    Ray ray = state.ray;
    HitRecord hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &state.rng;

    BSDFVal bsdf = mtl.getPlasticBSDF(ray, hit, rng);

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    // ------------------------------------------------------------
    // NEE / direct lighting
    // ------------------------------------------------------------
    if (gpuScene.nbLights > 0)
    {
        int lightIndex = int(rng->nextFloat() * gpuScene.nbLights);
        lightIndex = min(lightIndex, gpuScene.nbLights - 1);

        const Light& light = gpuScene.lights[lightIndex];
        LightSample ls = light.sample(hit.point, rng, gpuScene);

        if (ls.pdf > 0.f)
        {
            float3 shadowOrigin = hit.point + hit.normal * 1e-3f;
            Ray shadowRay(shadowOrigin, ls.direction, ray.time);

            if (!gpuScene.intersectAny(shadowRay, 1e-3f, ls.distance - 1e-3f))
            {
                float cosTheta = fmaxf(dot(hit.normal, ls.direction), 0.0f);

                if (cosTheta > 0.f)
                {
                    float3 f = mtl.evalPlasticBSDF(ray, hit, ls.direction);

                    float pdf_light = ls.pdf * (1.f / gpuScene.nbLights);
                    float pdf_bsdf  = mtl.plasticPDF(ray, hit, ls.direction);

                    float w = powerHeuristic(pdf_light, pdf_bsdf);

                    state.radiance +=
                        state.throughput *
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

    state.throughput =
        state.throughput *
        bsdf.brdf *
        cosTheta /
        bsdf.pdf;

    state.ray = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        ray.time
    );

    state.lastBounceWasDelta = false;
    state.lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeMirrorKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    HitRecord* hits,
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

    Ray ray = state.ray;
    HitRecord hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    BSDFVal bsdf = mtl.getMirrorBSDF(ray, hit);

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    state.throughput *= bsdf.brdf;

    state.ray = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        ray.time
    );

    state.lastBounceWasDelta = true;
    state.lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}

__global__
void shadeTransparentKernel(
    CudaScene& gpuScene,
    WavefrontState* states,
    HitRecord* hits,
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

    Ray ray = state.ray;
    HitRecord hit = hits[idx];
    Material mtl = gpuScene.materials[hit.materialIndex];

    RNG* rng = &state.rng;

    BSDFVal bsdf = mtl.getTransparentBSDF(
        ray,
        hit,
        rng,
        state.isInside
    );

    if (bsdf.pdf <= 1e-4f)
    {
        state.terminate();
        states[idx] = state;
        return;
    }

    state.throughput *= bsdf.brdf;

    state.ray = Ray(
        hit.point + bsdf.direction * 1e-3f,
        bsdf.direction,
        ray.time
    );

    state.lastBounceWasDelta = true;
    state.lastBsdfPdf = bsdf.pdf;

    states[idx] = state;

    atomicAdd(d_activeCount, 1);
}
*/