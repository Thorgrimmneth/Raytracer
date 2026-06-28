#include "shading_kernels.cuh"

__global__ void shadeMissKernel(float3 *origins, float3 *directions, float3 *throughput, float3 *accumBuffer,
                                int *pixelIndices, int missCount, bool safeSun)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= missCount)
        return;

    const float3 direction = directions[qid];

    if (direction.y <= 0.0f)
        return;

    const float3 origin = origins[qid];

    const float horizonFade = smoothstep(-0.05f, 0.02f, direction.y);

    const float segmentLength = sizeAtmosphere / skyColorSamples;

    //------------------------------------------------------------------
    // Phases
    //------------------------------------------------------------------

    const float mu = dot(direction, sunDirection);
    const float mu2Term = 1.0f + mu * mu;

    const float phaseR = 0.0596831f * mu2Term;

    const float temp = 1.5776f - 1.52f * mu;

    const float phaseM = 0.0195609427f * mu2Term * rsqrtf(temp) / temp;

    //------------------------------------------------------------------
    // Soleil : somme géométrique fermée
    //------------------------------------------------------------------

    const float sunSegmentLength = 15000.0f;

    const float qR = __expf(-hr * sunDirection.y * sunSegmentLength);

    const float qM = __expf(-hm * sunDirection.y * sunSegmentLength);

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

    float3 samplePosition = origin + direction * (0.5f * segmentLength);

    float height = fmaxf(samplePosition.y, 0.0f);

    float hrLocal = __expf(-height * hr);

    float hmLocal = __expf(-height * hm);

    const float rR = __expf(-hr * direction.y * segmentLength);

    const float rM = __expf(-hm * direction.y * segmentLength);

    for (int i = 0; i < skyColorSamples; ++i)
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
            -(betaR * (opticalDepthR + opticalDepthLightR) + betaM * (opticalDepthM + opticalDepthLightM));

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

    float3 sky = sumR * betaR * phaseR + sumM * betaM * phaseM * 0.3f;

    //------------------------------------------------------------------
    // Disque solaire
    //------------------------------------------------------------------

    const float cosTheta = dot(direction, sunDirection);

    const float sunDisk = smoothstep(cosSunAngularRadius, cosSunAngularRadiusHalf, cosTheta);

    const float t = clamp((sunDirection.y + 0.4f) / 1.4f, 0.0f, 1.0f);

    const float sunset = (1.0f - t) * (1.0f - t);

    float3 sunColor = lerp(make_float3(30.f, 27.f, 24.f), make_float3(60.f, 25.f, 10.f), sunset);

    if (!safeSun)
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

    accumBuffer[pixelIndices[qid]] += throughput[qid] * sky;
}

__global__ void shadeLambertKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *p_rng,
                                   float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices, int *pixelIndices,
                                   float3 *nextOrigins, float3 *nextDirections, float3 *nextThroughput,
                                   int *nextPixelIndices, bool *nextLastBounceWasDelta, float *nextLastBsdfPdf,
                                   bool *nextIsInside, RNG *nextRng, int *nextActiveCount, int activeCount,
                                   float3 *shadowOrigins, float3 *shadowDirections, float3 *shadowContributions,
                                   int *shadowPixelIndices, float *shadowMaxDistances, int *shadowCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    float3 throughput = throughputs[qid];
    int materialIndex = hitMaterialIndices[qid];
    float3 pos = hitPositions[qid];
    float3 normal = hitNormals[qid];

    Material &mtl = scene.materials[materialIndex];

    RNG &rng = p_rng[qid];
    BSDFVal bsdf;

    float e1 = rng.nextFloat();
    float e2 = rng.nextFloat();
    float r = sqrtf(e1);
    float phi = 2.f * GPUPIf * e2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.f, 1.f - x * x - y * y));
    float3 T, B;
    getTangentFrame(normal, T, B);

    bsdf.direction = x * T + y * B + z * normal;
    bsdf.pdf = fmaxf(dot(normal, bsdf.direction), 0.f) * GPUInvPIf;
    bsdf.brdf = mtl.color() * GPUInvPIf;
    bool alive = (bsdf.pdf > 1e-4f);
    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------
    int nbLights = scene.nbLights;
    bool neeAlive = alive && (nbLights > 0);
    float3 contribution = make_float3(0.f);
    float3 origin = pos + normal * 1e-3f;
    float3 directionToLight = make_float3(0.f);
    float maxDistance = 0.f;
    if (neeAlive)
    {
        int lightIndex = selectLightByImportance(nbLights, scene.lightCumulativeWeights, rng);

        float lightSelectionProb = getLightProbability(nbLights, scene.lightProbabilities, lightIndex);

        Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(pos, rng, scene);
        neeAlive = ls.pdf > 0.f;
        if (neeAlive)
        {
            float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);
            neeAlive = cosTheta > 0.f;
            if (neeAlive)
            {
                float3 direction = directions[qid];
                float3 f = mtl.evalLambertBSDF();
                float pdf_bsdf = mtl.lambertPDF(direction, normal, ls.direction);

                float pdf_light = ls.pdf * lightSelectionProb;
                neeAlive = pdf_light > 0.f;
                if (neeAlive)
                {
                    float w = powerHeuristic(pdf_light, pdf_bsdf);
                    maxDistance = ls.distance;
                    directionToLight = ls.direction;
                    contribution = throughput * f * ls.radiance * cosTheta * w / pdf_light;
                }
            }
        }
    }
    if (alive)
    {
        float cosTheta = fmaxf(dot(normal, bsdf.direction), 0.0f);

        throughput *= bsdf.brdf * cosTheta / bsdf.pdf;

        if (depth > 2)
        {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

            p = clamp(p, 0.1f, 1.f);

            if (rng.nextFloat() > p)
            {
                alive = false;
            }
            else
            {
                throughput /= p;
            }
        }
    }
    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------

    unsigned mask = __ballot_sync(0xffffffff, alive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(nextActiveCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!alive)
        return;

    int dst = warpBase + localRank;

    nextOrigins[dst] = pos + bsdf.direction * 1e-3f;

    nextDirections[dst] = bsdf.direction;

    nextThroughput[dst] = throughput;

    nextPixelIndices[dst] = pixelIndices[qid];

    nextLastBounceWasDelta[dst] = false;

    nextLastBsdfPdf[dst] = bsdf.pdf;

    nextIsInside[dst] = false;

    nextRng[dst] = rng;

    // nee queue compaction
    mask = __ballot_sync(0xffffffff, neeAlive);
    localRank = __popc(mask & ((1u << lane) - 1));
    warpCount = __popc(mask);
    if (lane == 0)
    {
        warpBase = atomicAdd(shadowCount, warpCount);
    }
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);
    if (!neeAlive)
        return;
    shadowOrigins[warpBase + localRank] = origin;
    shadowDirections[warpBase + localRank] = directionToLight;
    shadowContributions[warpBase + localRank] = contribution;
    shadowPixelIndices[warpBase + localRank] = pixelIndices[qid];
    shadowMaxDistances[warpBase + localRank] = maxDistance;
}

__global__ void shadeMetalKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                 float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices, int *pixelIndices,
                                 float3 *accumBuffer, float3 *nextOrigins, float3 *nextDirections,
                                 float3 *nextThroughput, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                                 float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRng, int *nextActiveCount,
                                 int activeCount, float3 *shadowOrigins, float3 *shadowDirections,
                                 float3 *shadowContributions, int *shadowPixelIndices, float *shadowMaxDistances,
                                 int *shadowCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    float3 throughput = throughputs[qid];

    float3 direction = directions[qid];

    int materialIndex = hitMaterialIndices[qid];
    float3 pos = hitPositions[qid];

    Material &mtl = scene.materials[materialIndex];

    RNG &rng = rngs[qid];
    float3 normal = hitNormals[qid];
    BSDFVal bsdf = mtl.getMetalBSDF(direction, normal, rng);

    float cosTheta = fmaxf(dot(normal, bsdf.direction), 0.0f);

    bool alive = (bsdf.pdf > 1e-4f);

    // ------------------------------------------------------------
    // DIRECT LIGHTING / NEE
    // Seulement pour les matériaux non-delta.
    // ------------------------------------------------------------

    int nbLights = scene.nbLights;
    bool neeAlive = alive && (nbLights > 0);
    float3 contribution = make_float3(0.f);
    float3 origin = pos + normal * 1e-3f;
    float3 directionToLight = make_float3(0.f);
    float maxDistance = 0.f;
    if (neeAlive)
    {
        int lightIndex = selectLightByImportance(nbLights, scene.lightCumulativeWeights, rng);

        float lightSelectionProb = getLightProbability(nbLights, scene.lightProbabilities, lightIndex);

        Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(pos, rng, scene);
        neeAlive = ls.pdf > 0.f;
        if (neeAlive)
        {
            float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);
            neeAlive = cosTheta > 0.f;
            if (neeAlive)
            {
                float3 f = mtl.evalMetalBSDF(direction, normal, ls.direction);
                float pdf_bsdf = mtl.metalPDF(direction, normal, ls.direction);

                float pdf_light = ls.pdf * lightSelectionProb;
                neeAlive = pdf_light > 0.f;
                if (neeAlive)
                {
                    float w = powerHeuristic(pdf_light, pdf_bsdf);
                    maxDistance = ls.distance;
                    directionToLight = ls.direction;
                    contribution += throughput * f * ls.radiance * cosTheta * w / pdf_light;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------
    if (alive)
    {
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
                alive = false;
            }
            else
            {
                throughput /= p;
            }
        }
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------

    unsigned mask = __ballot_sync(0xffffffff, alive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(nextActiveCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!alive)
        return;

    int dst = warpBase + localRank;

    nextOrigins[dst] = pos + bsdf.direction * 1e-3f;
    nextDirections[dst] = bsdf.direction;
    nextThroughput[dst] = throughput;

    nextPixelIndices[dst] = pixelIndices[qid];

    nextLastBounceWasDelta[dst] = false;
    nextLastBsdfPdf[dst] = bsdf.pdf;
    nextIsInside[dst] = false;

    nextRng[dst] = rng;

    // nee queue compaction
    mask = __ballot_sync(0xffffffff, neeAlive);
    localRank = __popc(mask & ((1u << lane) - 1));
    warpCount = __popc(mask);
    if (lane == 0)
    {
        warpBase = atomicAdd(shadowCount, warpCount);
    }
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);
    if (!neeAlive)
        return;
    shadowOrigins[warpBase + localRank] = origin;
    shadowDirections[warpBase + localRank] = directionToLight;
    shadowContributions[warpBase + localRank] = contribution;
    shadowPixelIndices[warpBase + localRank] = pixelIndices[qid];
    shadowMaxDistances[warpBase + localRank] = maxDistance;
    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------
}

__global__ void shadePlasticNEEKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                      float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices,
                                      int *pixelIndices, float3 *accumBuffer, int nbLights, float *lightWeights,
                                      int activeCount, float3 *shadowOrigins, float3 *shadowDirections,
                                      float3 *shadowContributions, int *shadowPixelIndices, float *shadowMaxDistances,
                                      int *shadowCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount || nbLights <= 0) // Early exit for empty light lists
        return;

    float3 pos = hitPositions[qid];
    float3 normal = hitNormals[qid];
    int materialIndex = hitMaterialIndices[qid];
    Material &mtl = scene.materials[materialIndex];
    RNG &rng = rngs[qid];

    bool neeAlive = scene.nbLights > 0;

    float3 contribution = make_float3(0.f);
    float3 directionToLight = make_float3(0.f);
    float maxDistance = 0.0f;

    if (neeAlive)
    {
        int lightIndex = selectLightByImportance(nbLights, scene.lightCumulativeWeights, rng);

        float lightSelectionProb = getLightProbability(nbLights, scene.lightProbabilities, lightIndex);

        Light &light = scene.lights[lightIndex];

        LightSample ls = light.sample(pos, rng, scene);

        neeAlive = ls.pdf > 0.f;

        if (neeAlive)
        {
            float cosTheta = fmaxf(dot(normal, ls.direction), 0.0f);

            neeAlive = cosTheta > 0.f;

            if (neeAlive)
            {
                float3 direction = directions[qid];

                float3 f = mtl.evalPlasticBSDF(direction, normal, ls.direction);

                neeAlive = (f.x > 0.f || f.y > 0.f || f.z > 0.f);

                if (neeAlive)
                {
                    float pdf_bsdf = mtl.plasticPDF(direction, normal, ls.direction);

                    float pdf_light = ls.pdf * lightSelectionProb;

                    neeAlive = pdf_light > 0.f;

                    if (neeAlive)
                    {
                        float w = powerHeuristic(pdf_light, pdf_bsdf);

                        directionToLight = ls.direction;
                        maxDistance = ls.distance - 1e-3f;

                        contribution = throughputs[qid] * f * ls.radiance * cosTheta * w / pdf_light;
                    }
                }
            }
        }
    }

    unsigned mask = __ballot_sync(0xffffffff, neeAlive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(shadowCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!neeAlive)
        return;

    int dst = warpBase + localRank;

    shadowOrigins[dst] = pos + normal * 1e-3f;
    shadowDirections[dst] = directionToLight;
    shadowMaxDistances[dst] = maxDistance;
    shadowContributions[dst] = contribution;
    shadowPixelIndices[dst] = pixelIndices[qid];
}

__global__ void shadePlasticKernel(Material *materials, float3 *directions, float3 *throughputs, RNG *rngs,
                                   float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices, int *pixelIndices,
                                   float3 *accumBuffer, float3 *nextOrigins, float3 *nextDirections,
                                   float3 *nextThroughput, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                                   float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRng, int *nextActiveCount,
                                   int activeCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= activeCount)
        return;

    int matIdx = hitMaterialIndices[qid];
    RNG rng = rngs[qid];

    // Extraire composantes (le compilateur optimise ça en registres)
    float3 normal = hitNormals[qid];
    float3 dir = directions[qid];
    float3 thru = throughputs[qid];

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
    float e1 = rng.nextFloat();
    float e2 = rng.nextFloat();

    // Tangent frame (optimisé avec branchement minimal)
    float3 T, B;
    getTangentFrame(normal, T, B); // Fonction inline

    float3 newDir;
    float3 brdf;
    float pdf;
    float cosThetaNew;

    if (rng.nextFloat() < specW)
    {
        // SPECULAR (GGX)
        // Éviter d'allouer V en registre - calculer directement
        float Vx = dot(wo, T);
        float Vy = dot(wo, B);
        float Vz = dot(wo, normal);

        // Stretch
        Vx *= alpha;
        Vy *= alpha; // Vz reste

        // Sample GGX
        float3 h = sampleGGX(e1, e2, Vx, Vy, Vz); // Retourne normalité

        // Transform back
        float3 H = h.x * T + h.y * B + h.z * normal;
        float HdotV = dot(H, wo);
        newDir = reflect(-wo, H);
        cosThetaNew = dot(normal, newDir);
        if (cosThetaNew <= 0.f)
        {
            pdf = 0.f;
            brdf = make_float3(0.f);
        }
        else
        {
            // Évaluer GGX sans redondance
            float NdotH = dot(normal, H);
            float NdotV = cosThetaView;
            float NdotL = cosThetaNew;

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
            pdf = (HdotV > 0.f && NdotV > 0.f) ? G1V * D / fmaxf(4.f * NdotV, 1e-8f) : 0.f;
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
        float s, c;
        sincosf(phi, &s, &c);
        newDir = sint * c * T + sint * s * B + cost * normal;
        brdf = baseColor * GPUInvPIf * (1.f - specW);
        cosThetaNew = dot(normal, newDir);
        pdf = (1.f - specW) * fmaxf(cosThetaNew, 0.f) * GPUInvPIf;
    }

    // === EARLY EXIT ===
    bool alive = (pdf > 1e-4f);

    // === UPDATE THROUGHPUT ===
    if (alive)
    {
        float cosTheta = fmaxf(cosThetaNew, 0.f);
        thru *= brdf * cosTheta / pdf;

        // === RUSSIAN ROULETTE ===
        if (depth > 2)
        {
            float p = fmaxf(thru.x, fmaxf(thru.y, thru.z));
            p = clamp(p, 0.1f, 1.f);
            if (rng.nextFloat() > p)
            {
                alive = false;
            }
            else
            {
                thru /= p;
            }
        }
    }

    unsigned mask = __ballot_sync(0xffffffff, alive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(nextActiveCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!alive)
        return;

    int dst = warpBase + localRank;

    nextOrigins[dst] = hitPositions[qid] + newDir * 1e-3f;
    nextDirections[dst] = newDir;
    nextThroughput[dst] = thru;

    nextPixelIndices[dst] = pixelIndices[qid];

    nextLastBounceWasDelta[dst] = false;
    nextLastBsdfPdf[dst] = pdf;
    nextIsInside[dst] = false;

    nextRng[dst] = rng;

    // === QUEUE DISPATCH ===
}

__global__ void shadeMirrorKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                  float3 *hitPositions, float3 *hitNormals, int *hitMaterialIndices, int *pixelIndices,
                                  float3 *accumBuffer, float3 *nextOrigins, float3 *nextDirections,
                                  float3 *nextThroughput, int *nextPixelIndices, bool *nextLastBounceWasDelta,
                                  float *nextLastBsdfPdf, bool *nextIsInside, RNG *nextRng, int *nextActiveCount,
                                  int activeCount, uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    float3 throughput = throughputs[qid];

    float3 direction = directions[qid];

    int materialIndex = hitMaterialIndices[qid];
    float3 pos = hitPositions[qid];

    Material &mtl = scene.materials[materialIndex];

    float3 normal = hitNormals[qid];
    BSDFVal bsdf = mtl.getMirrorBSDF(direction, normal);

    bool alive = (bsdf.pdf > 1e-4f);

    // ------------------------------------------------------------
    // UPDATE THROUGHPUT
    // ------------------------------------------------------------

    throughput *= bsdf.brdf;

    // ------------------------------------------------------------
    // RUSSIAN ROULETTE
    // ------------------------------------------------------------

    if (depth > 2)
    {
        RNG &rng = rngs[qid];
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = clamp(p, 0.1f, 1.f);

        if (rng.nextFloat() > p)
        {
            alive = false;
        }
        else
        {
            throughput /= p;
        }
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    unsigned mask = __ballot_sync(0xffffffff, alive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(nextActiveCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!alive)
        return;

    int dst = warpBase + localRank;

    nextOrigins[dst] = pos + bsdf.direction * 1e-3f;
    nextDirections[dst] = bsdf.direction;
    nextThroughput[dst] = throughput;

    nextPixelIndices[dst] = pixelIndices[qid];

    nextLastBounceWasDelta[dst] = true;
    nextLastBsdfPdf[dst] = bsdf.pdf;
    nextIsInside[dst] = false;

    nextRng[dst] = rngs[qid];

    // ------------------------------------------------------------
    // COMPACT NEXT ACTIVE QUEUE
    // ------------------------------------------------------------
}

__global__ void shadeTransparentKernel(CudaScene scene, float3 *directions, float3 *throughputs, RNG *rngs,
                                       bool *isInsides, float3 *hitPositions, float3 *hitNormals,
                                       int *hitMaterialIndices, int *pixelIndices, float3 *accumBuffer,
                                       float3 *nextOrigins, float3 *nextDirections, float3 *nextThroughput,
                                       int *nextPixelIndices, bool *nextLastBounceWasDelta, float *nextLastBsdfPdf,
                                       bool *nextIsInside, RNG *nextRng, int *nextActiveCount, int activeCount,
                                       uint depth)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    float3 throughput = throughputs[qid];

    bool &isInside = isInsides[qid];

    int materialIndex = hitMaterialIndices[qid];
    float3 pos = hitPositions[qid];
    float3 normal = hitNormals[qid];

    Material &mtl = scene.materials[materialIndex];

    BSDFVal bsdf = mtl.getTransparentBSDF(directions[qid], normal, rngs[qid], isInside);

    bool alive = (bsdf.pdf > 1e-4f);

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

        RNG &rng = rngs[qid];
        if (rng.nextFloat() > p)
        {
            alive = false;
        }
        else
        {
            throughput /= p;
        }
    }

    // ------------------------------------------------------------
    // NEXT origin, direction
    // ------------------------------------------------------------
    unsigned mask = __ballot_sync(0xffffffff, alive);

    int lane = threadIdx.x & 31;

    int localRank = __popc(mask & ((1u << lane) - 1));

    int warpCount = __popc(mask);

    int warpBase = 0;

    if (lane == 0)
    {
        warpBase = atomicAdd(nextActiveCount, warpCount);
    }

    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    if (!alive)
        return;

    int dst = warpBase + localRank;

    nextOrigins[dst] = pos + bsdf.direction * 1e-3f;
    nextDirections[dst] = bsdf.direction;
    nextThroughput[dst] = throughput;

    nextPixelIndices[dst] = pixelIndices[qid];

    nextLastBounceWasDelta[dst] = true;
    nextLastBsdfPdf[dst] = bsdf.pdf;
    nextIsInside[dst] = isInside;

    nextRng[dst] = rngs[qid];
}

__global__ void shadeEmissiveKernel(CudaScene scene, float3 *origins, float3 *directions, float3 *throughputs,
                                    bool *lastBounceWasDelta, float *lastBsdfPdf, int *hitMaterialIndices,
                                    int *pixelIndices, float3 *accumBuffer, int activeCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    float3 throughput = throughputs[qid];

    // ------------------------------------------------------------
    // HIT
    // ------------------------------------------------------------

    int materialIndex = hitMaterialIndices[qid];
    Material &mtl = scene.materials[materialIndex];

    // ------------------------------------------------------------
    // EMISSIVE
    // ------------------------------------------------------------

    float3 emission = mtl.color() * mtl.intensity();

    if (lastBounceWasDelta[qid])
    {
        accumBuffer[pixelIndices[qid]] += throughput * emission;
    }
    else
    {
        float lightPdf = scene.lightPdf(origins[qid], directions[qid]);

        float w = powerHeuristic(lastBsdfPdf[qid], lightPdf);

        accumBuffer[pixelIndices[qid]] += throughput * emission * w;
    }

    return;
}