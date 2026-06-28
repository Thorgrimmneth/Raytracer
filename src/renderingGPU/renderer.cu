#include "renderer.hpp"

#include "../../../devicePrograms/launch_radiance_params.cuh"
#include "../../../devicePrograms/launch_shadow_params.cuh"
#include "camera/camera.cuh"
#include "scene/scene.cuh"

#include "integrators/pathtracer_integrator.cuh"

#include "../utils/definesCPU.hpp"
#include "renderingUtils/post_treatment.cuh"
#include "renderingUtils/shading_kernels.cuh"
#include "utils/constant.cuh"
#include "utils/fillBuffers.cuh"
#include "utils/shadingData.cuh"
#include "utils/shadingLaunch.cuh"
#include "utils/shadowData.cuh"
#include "utils/simplifiedDef.cuh"
#include "utils/sortQueues.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <thrust/sort.h>

__constant__ int nbBounces;
__constant__ float earthRadius;
__constant__ float3 sunDirection;
__constant__ int skyColorSamples;
__constant__ float hr;
__constant__ float hm;
__constant__ float3 betaR;
__constant__ float3 betaM;
__constant__ float exposure;
__constant__ Camera camera;
__constant__ float sizeAtmosphere;
__constant__ float cosSunAngularRadius;
__constant__ float cosSunAngularRadiusHalf;

enum class RenderMode
{
    Megakernel,
    Wavefront
};

class Renderer::Impl
{
  public:
    RenderMode renderMode = RenderMode::Wavefront;

    int width = 1920;
    int height = 1080;
    int minNumberShadowRay = width * height;
    Camera h_camera;

    float threshold = 1.0f;
    float bloomStrength = 0.25f;
    float exposure = 1.0f;

    int sampleCount = 0;
    int maxBounces = 5;

    size_t hdrBufferSize = 0;

    float value = 1.f;
    float *d_value = nullptr;

    CudaScene gpuScene;

    uint depth = 0;

    // Buffers utilisés par le renderer mais pas directement par les shaders
    float3 *d_normalizedBuffer = nullptr;
    float3 *d_hdrBloomBuffer = nullptr;
    float3 *d_bloomBuffer = nullptr;
    float3 *d_tempBuffer = nullptr;
    float3 *d_convergenceBuffer = nullptr;
    float3 *d_accumBuffer = nullptr;

    RayQueue currentQueue;
    RayQueue nextQueue;
    SortedRayQueue sortedQueue;
    ShadowRayQueue shadowQueue;
    int *d_shadowCount = nullptr;
    HitBuffers hitBuffers;
    bool launchedShadeKernel = true;
    // Buffers intermédiaires pour le ray sorting
    int *d_keys = nullptr;
    int *d_values = nullptr;
    MaterialRanges *d_ranges = nullptr;
    unsigned int *d_octantKeys = nullptr;

    dim3 blockSize = dim3(16, 16);
    dim3 gridSize;

    cudaStream_t stream = nullptr;

    int w1 = 0;
    int h1 = 0;

    int w2 = 0;
    int h2 = 0;

    float3 *d_lvl1 = nullptr;
    float3 *d_lvl2 = nullptr;

    cudaGraphicsResource *cudaTextureResource = nullptr;
};

Renderer::Renderer() { impl = new Impl(); }

Renderer::~Renderer()
{
    cleanUp();
    delete impl;
}

// Helper

void Renderer::setInteropResource(cudaGraphicsResource *resource) { impl->cudaTextureResource = resource; }

void Renderer::changeMode()
{
    switch (impl->renderMode)
    {
    case RenderMode::Megakernel:
        impl->renderMode = RenderMode::Wavefront;
        std::cout << "Render mode: Wavefront" << std::endl;
        break;
    case RenderMode::Wavefront:
        impl->renderMode = RenderMode::Megakernel;
        std::cout << "Render mode: Megakernel" << std::endl;
        break;
    }
}

void Renderer::resetAccumulation()
{
    impl->sampleCount = 0;

    cudaMemset(impl->d_accumBuffer, 0, impl->hdrBufferSize);
}

int Renderer::getFrameNumber() { return impl->sampleCount; }

HOST Camera initCamera(int width, int height)
{
    // ===== Camera =====
    float3 camPos = make_float3(8.f, 2.f, 3.f);
    float3 camTarget = make_float3(0.f, 0.f, 0.f);
    float3 camUp = make_float3(0.f, 1.f, 0.f);

    float fov = 60.f;
    float aspect = (float)width / (float)height;
    float focalDistance = 1.f;

    // === Base vectors EXACTEMENT comme CPU ===
    float3 w = normalize(camPos - camTarget);
    float3 u = normalize(cross(camUp, w));
    float3 v = normalize(cross(w, u));

    // === Viewport ===
    float theta = fov * 3.14159265f / 180.f;
    float viewportHeight = 2.f * tanf(theta * 0.5f) * focalDistance;
    float viewportWidth = viewportHeight * aspect;

    float3 viewportU = u * viewportWidth;
    float3 viewportV = v * viewportHeight;

    float3 topLeft = camPos - w * focalDistance + viewportV * 0.5f - viewportU * 0.5f;
    Camera camera = Camera{camPos, topLeft, viewportU, viewportV};
    return camera;
}

HOST void initConstant(int width, int height, Camera c_camera, float4 sunDir)
{
    int c_nbBounces = 8;
    float c_earthRadius = 6360e3f;
    float3 c_sunDirection = make_float3(sunDir);
    int c_skyColorSamples = 4;
    float c_hr = 1.f / 7994.f;
    float c_hm = 1.f / 1200.f;
    float3 c_betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    float3 c_betaM = make_float3(21e-6f);
    float c_exposure = 1.f;
    float c_sizeAtmosphere = 60000.f;
    float sunAngularRadius = 2.1f * GPUPIf / 180.f;
    float c_sunAngularRadius = cosf(sunAngularRadius);
    float c_sunAngularRadiusHalf = cosf(sunAngularRadius * 0.5f);
    cudaMemcpyToSymbol(nbBounces, &c_nbBounces, sizeof(int));
    cudaMemcpyToSymbol(earthRadius, &c_earthRadius, sizeof(float));
    cudaMemcpyToSymbol(sunDirection, &c_sunDirection, sizeof(float3));
    cudaMemcpyToSymbol(skyColorSamples, &c_skyColorSamples, sizeof(int));
    cudaMemcpyToSymbol(hr, &c_hr, sizeof(float));
    cudaMemcpyToSymbol(hm, &c_hm, sizeof(float));
    cudaMemcpyToSymbol(betaR, &c_betaR, sizeof(float3));
    cudaMemcpyToSymbol(betaM, &c_betaM, sizeof(float3));
    cudaMemcpyToSymbol(exposure, &c_exposure, sizeof(float));
    cudaMemcpyToSymbol(camera, &c_camera, sizeof(Camera));
    cudaMemcpyToSymbol(sizeAtmosphere, &c_sizeAtmosphere, sizeof(float));
    cudaMemcpyToSymbol(cosSunAngularRadius, &c_sunAngularRadius, sizeof(float));
    cudaMemcpyToSymbol(cosSunAngularRadiusHalf, &c_sunAngularRadiusHalf, sizeof(float));
}

void Renderer::init(int p_width, int p_height, float sunDirx, float sunDiry, float sunDirz, int rngManip)
{
    impl->h_camera = initCamera(p_width, p_height);
    initConstant(p_width, p_height, impl->h_camera, make_float4(sunDirx, sunDiry, sunDirz, 0.f));
    impl->width = p_width;
    impl->height = p_height;
    impl->minNumberShadowRay = p_width * p_height;
    float4 sunDir = make_float4(sunDirx, sunDiry, sunDirz, 0.f);

    setSeed(43);

    // impl->gpuScene = spheresScene(sunDir);
    // impl->gpuScene = implicitSpheresScene(sunDir);
    impl->gpuScene = singleObject(sunDir, rngManip);
    impl->hdrBufferSize = impl->width * impl->height * sizeof(float3);

    // =========================
    // GPU buffers
    // =========================

    cudaMalloc(&impl->d_normalizedBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_bloomBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_hdrBloomBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_tempBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_accumBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_convergenceBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_value, sizeof(float));
    cudaMalloc(&impl->d_ranges, sizeof(MaterialRanges));

    size_t pixelCount = impl->width * impl->height;

    cudaMalloc(&impl->d_keys, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_values, pixelCount * sizeof(int));

    cudaMalloc(&impl->d_octantKeys, pixelCount * sizeof(unsigned int));

    impl->currentQueue.init(pixelCount);
    impl->nextQueue.init(pixelCount);
    impl->sortedQueue.init(pixelCount);
    impl->hitBuffers.init(pixelCount);
    impl->shadowQueue.init(pixelCount * 2); // Assuming a maximum of 2 shadow rays per pixel
    cudaMalloc(&impl->d_shadowCount, sizeof(int));

    cudaMemcpy(impl->currentQueue.activeCount, &pixelCount, sizeof(int), cudaMemcpyHostToDevice);

    // =========================
    // CUDA Stream for async operations
    // =========================

    cudaStreamCreate(&impl->stream);

    // =========================

    impl->blockSize = dim3(16, 16);

    impl->gridSize = dim3((impl->width + impl->blockSize.x - 1) / impl->blockSize.x,
                          (impl->height + impl->blockSize.y - 1) / impl->blockSize.y);

    // =========================
    // Bloom mip chain
    // =========================

    impl->w1 = impl->width / 2;
    impl->h1 = impl->height / 2;

    impl->w2 = impl->w1 / 2;
    impl->h2 = impl->h1 / 2;

    cudaMalloc(&impl->d_lvl1, impl->w1 * impl->h1 * sizeof(float3));

    cudaMalloc(&impl->d_lvl2, impl->w2 * impl->h2 * sizeof(float3));

    impl->sampleCount = 0;
}

GLOBAL
void compareBuffers(const float3 *d_currentBuffer, const float3 *d_previousBuffer, int width, int height, float *value)
{
    // Block-level reduction without atomic operations for better performance
    __shared__ float blockSum;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        blockSum = 0.0f;

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float localSum = 0.0f;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        float3 current = d_currentBuffer[idx];
        float3 previous = d_previousBuffer[idx];
        float3 diff = abs(current - previous);
        localSum = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);

    // Write warp result to shared memory
    if (threadIdx.x % warpSize == 0)
        atomicAdd(&blockSum, localSum);

    __syncthreads();

    // One thread writes block result
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(value, blockSum);
}

// MEGAKERNEL
GLOBAL
void renderKernel(CudaScene gpuScene, float3 *d_accumBuffer, int width, int height, int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = y * width + x;
    uint seed = (pixelIndex * 0x9E3779B9u) ^ (sampleCount * 0x6C078965u);
    RNG localState(seed);

    float3 finalColor = make_float3(0.f);

    float sx = (x + localState.nextFloat()) / (float)(width - 1);
    float sy = (y + localState.nextFloat()) / (float)(height - 1);

    float3 rayTarget = camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV;
    float3 direction = normalize(rayTarget - camera.cameraPos);

    finalColor += PathtracerIntegrator::lighting(gpuScene, camera.cameraPos, direction, 0, 1e20f, localState);

    d_accumBuffer[pixelIndex] += finalColor;
}

// WAVEFRONT INIT
GLOBAL
void generatePrimaryRaysKernel(float3 *directions, RNG *p_rng, int *p_pixelIndices, int width, int height,
                               int sampleCount, float invWidth, float invHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = y * width + x;
    p_pixelIndices[pixelIndex] = pixelIndex;

    uint seed = (pixelIndex * 0x9E3779B9u) ^ (sampleCount * 0x6C078965u);
    RNG rng(seed);

    float sx = (x + rng.nextFloat()) * invWidth;
    float sy = (y + rng.nextFloat()) * invHeight;
    p_rng[pixelIndex] = rng;
    float3 rayTarget = camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV;
    directions[pixelIndex] = normalize(rayTarget - camera.cameraPos);
}

void Renderer::applyBloom()
{
    // =========================
    // Extracts bright pixels, also normalizes
    // =========================

    extractBright<<<impl->gridSize, impl->blockSize>>>(impl->d_accumBuffer, impl->d_normalizedBuffer,
                                                       impl->d_bloomBuffer, impl->width, impl->height, impl->threshold,
                                                       1.f / (float)impl->sampleCount);

    // =========================
    // Blur bloom
    // =========================

    applyMultiScaleBloom(impl->d_bloomBuffer, impl->d_tempBuffer, impl->d_lvl1, impl->d_lvl2, impl->w1, impl->h1,
                         impl->w2, impl->h2, impl->width, impl->height);
}

float Renderer::renderFrame(bool outputImage, bool convergence)
{
    renderKernel<<<impl->gridSize, impl->blockSize>>>(impl->gpuScene, impl->d_accumBuffer, impl->width, impl->height,
                                                      impl->sampleCount);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "renderKernel error: " << cudaGetErrorString(err) << std::endl;
    }

    impl->sampleCount++;

    applyBloom();

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compareBuffers<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_convergenceBuffer,
                                                            impl->width, impl->height, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergenceBuffer, impl->d_normalizedBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
    }

    if (!outputImage)
        return impl->value;

    cudaGraphicsMapResources(1, &impl->cudaTextureResource);

    cudaArray_t textureArray;

    cudaGraphicsSubResourceGetMappedArray(&textureArray, impl->cudaTextureResource, 0, 0);

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = textureArray;

    cudaSurfaceObject_t surface = 0;

    cudaCreateSurfaceObject(&surface, &desc);

    finalizeImageV2<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_bloomBuffer,
                                                         impl->d_hdrBloomBuffer, surface, impl->width, impl->height,
                                                         impl->exposure, impl->bloomStrength);

    cudaDestroySurfaceObject(surface);

    cudaGraphicsUnmapResources(1, &impl->cudaTextureResource);

    return impl->value;
}

__global__ void accumulateShadowKernel(const float3 *shadowContributions, const float3 *shadowTransmittance,
                                       const int *shadowPixelIndices, float3 *accumBuffer, int shadowCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= shadowCount)
        return;

    accumBuffer[shadowPixelIndices[idx]] += shadowContributions[idx] * shadowTransmittance[idx];
}

float Renderer::renderFrameWavefront(bool outputImage, bool convergence)
{
    int pixelCount = impl->width * impl->height;

    dim3 block2D = impl->blockSize;
    dim3 grid2D = impl->gridSize;

    dim3 block1D(256);

    auto gridForCount = [](int count) -> dim3 { return dim3((count + 255) / 256); };

    cudaError_t err;

    // -------------------------------------------------------------------------
    // 1. Génération des rayons primaires + activeQueue
    // -------------------------------------------------------------------------
    float invWidth = 1.f / (float)(impl->width - 1);
    float invHeight = 1.f / (float)(impl->height - 1);
    generatePrimaryRaysKernel<<<grid2D, block2D>>>(impl->currentQueue.directions, impl->currentQueue.rng,
                                                   impl->currentQueue.pixelIndices, impl->width, impl->height,
                                                   impl->sampleCount, invWidth, invHeight);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "generatePrimaryRaysKernel error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    int threads = 256;
    int blocks = (pixelCount + threads - 1) / threads;

    initFloat3Buffer<<<blocks, threads>>>(impl->currentQueue.origins, pixelCount, impl->h_camera.cameraPos);

    initFloat3Buffer<<<blocks, threads>>>(impl->currentQueue.throughputs, pixelCount, make_float3(1.f));

    initBoolBuffer<<<blocks, threads>>>(impl->currentQueue.isInside, pixelCount, false);

    initBoolBuffer<<<blocks, threads>>>(impl->currentQueue.lastBounceWasDelta, pixelCount, false);

    initFloatBuffer<<<blocks, threads>>>(impl->currentQueue.lastBsdfPdf, pixelCount, 1.f);

    initIntBuffer<<<blocks, threads>>>(impl->d_keys, pixelCount, (int)MISS);
    initIntBuffer<<<blocks, threads>>>(impl->d_values, pixelCount, 0);
    int h_activeCount = pixelCount;
    int h_shadowCount = 0;
    impl->gpuScene.shadowPass.params.materials = impl->gpuScene.materials;
    impl->gpuScene.shadowPass.params.nbMaterials = impl->gpuScene.nbMaterials;
    // -------------------------------------------------------------------------
    // 2. Boucle wavefront activeQueue
    // -------------------------------------------------------------------------
    for (int bounce = 0; bounce < impl->maxBounces; bounce++)
    {

        if (h_activeCount == 0)
            break;

        cudaMemset(impl->nextQueue.activeCount, 0, sizeof(int));
        if(impl->launchedShadeKernel) cudaMemset(impl->d_shadowCount, 0, sizeof(int));
        // ---------------------------------------------------------------------
        // 2.1 Intersection uniquement des rayons actifs
        // ---------------------------------------------------------------------
        impl->gpuScene.radiancePass.params.origins = impl->currentQueue.origins;
        impl->gpuScene.radiancePass.params.directions = impl->currentQueue.directions;
        impl->gpuScene.radiancePass.params.hitPositions = impl->hitBuffers.positions;
        impl->gpuScene.radiancePass.params.hitNormals = impl->hitBuffers.normals;
        impl->gpuScene.radiancePass.params.hitMaterialIndices = impl->hitBuffers.materialIndices;
        impl->gpuScene.radiancePass.params.hitMask = impl->hitBuffers.mask;
        impl->gpuScene.radiancePass.params.activeCount = h_activeCount;

        cudaMemcpy(reinterpret_cast<void *>(impl->gpuScene.radiancePass.d_params),
                   &impl->gpuScene.radiancePass.params, sizeof(LaunchRadianceParams),
                   cudaMemcpyHostToDevice);
        OPTIX_CHECK(optixLaunch(impl->gpuScene.radiancePass.pipeline,
                                0, // stream
                                impl->gpuScene.radiancePass.d_params, sizeof(LaunchRadianceParams),
                                &impl->gpuScene.radiancePass.sbt, h_activeCount, 1, 1));

        classifyPairs<<<gridForCount(h_activeCount), block1D>>>(impl->gpuScene.materials, h_activeCount, impl->d_keys,
                                                                impl->d_values, impl->hitBuffers.mask,
                                                                impl->hitBuffers.materialIndices);
        thrust::sort_by_key(thrust::device, impl->d_keys, impl->d_keys + h_activeCount, impl->d_values);

        computeMaterialRanges<<<1, MATERIAL_TYPE_COUNT>>>(impl->d_keys, h_activeCount, impl->d_ranges);

        MaterialRanges ranges;

        cudaMemcpy(&ranges, impl->d_ranges, sizeof(MaterialRanges), cudaMemcpyDeviceToHost);

        reorderPaths<<<gridForCount(h_activeCount), block1D>>>(impl->d_values, impl->currentQueue, impl->sortedQueue,
                                                               impl->hitBuffers.positions, impl->hitBuffers.normals,
                                                               impl->hitBuffers.materialIndices, h_activeCount);
        for (int i = 0; i < MATERIAL_TYPE_COUNT; i++)
        {
            int offset = ranges.offset[i];
            int count = ranges.count[i];
            if (count > 0)
            {
                launchShadeKernel(static_cast<MaterialType>(i), impl->sortedQueue, impl->nextQueue, impl->gpuScene,
                                  impl->d_accumBuffer, offset, count, impl->shadowQueue, impl->d_shadowCount, bounce);
            }
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "shadeWavefrontKernel error: " << cudaGetErrorString(err) << std::endl;
            return -1.f;
        }

        // ---------------------------------------------------------------------
        // 2.3 Récupération du nombre de rayons actifs pour le prochain bounce
        // ---------------------------------------------------------------------

        cudaMemcpy(&h_activeCount, impl->nextQueue.activeCount, sizeof(int), cudaMemcpyDeviceToHost);

        std::swap(impl->currentQueue, impl->nextQueue);
        cudaMemcpy(&h_shadowCount, impl->d_shadowCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_shadowCount < impl->minNumberShadowRay)
        {
            impl->launchedShadeKernel = false;
            continue;
        }
        impl->gpuScene.shadowPass.params.origins = impl->shadowQueue.origins;
        impl->gpuScene.shadowPass.params.directions = impl->shadowQueue.directions;
        impl->gpuScene.shadowPass.params.maxDistances = impl->shadowQueue.maxDistances;
        impl->gpuScene.shadowPass.params.transmittance = impl->shadowQueue.transmittance;

        impl->gpuScene.shadowPass.params.activeCount = h_shadowCount;
        cudaMemcpy(reinterpret_cast<void *>(impl->gpuScene.shadowPass.d_params),
                   &impl->gpuScene.shadowPass.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice);

        OPTIX_CHECK(optixLaunch(impl->gpuScene.shadowPass.pipeline,
                                0, // stream
                                impl->gpuScene.shadowPass.d_params, sizeof(LaunchShadowParams),
                                &impl->gpuScene.shadowPass.sbt, h_shadowCount, 1, 1));
        accumulateShadowKernel<<<gridForCount(h_shadowCount), block1D>>>(
            impl->shadowQueue.contributions, impl->shadowQueue.transmittance, impl->shadowQueue.pixelIndices,
            impl->d_accumBuffer, h_shadowCount);
        impl->launchedShadeKernel = true;
        // sorting rays
        /*buildOctantKeys<<<gridForCount(h_activeCount), block1D>>>(impl->d_directions, h_activeCount,
        impl->d_octantKeys, impl->d_values); thrust::sort_by_key(thrust::device, impl->d_octantKeys, impl->d_octantKeys
        + h_activeCount, impl->d_values);

        reorderPaths<<<gridForCount(h_activeCount), block1D>>>(
            impl->d_values, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_hitPositions,
            impl->d_hitNormals, impl->d_hitMaterialIndices, impl->d_pixelIndices, impl->d_lastBounceWasDelta,
            impl->d_lastBsdfPdf, impl->d_isInside, impl->d_rng, impl->d_sortedOrigins, impl->d_sortedDirections,
            impl->d_sortedThroughput, impl->d_sortedHitPositions, impl->d_sortedHitNormals,
            impl->d_sortedHitMaterialIndices, impl->d_sortedPixelIndices, impl->d_sortedLastBounceWasDelta,
            impl->d_sortedLastBsdfPdf, impl->d_sortedIsInside, impl->d_sortedRNG, h_activeCount);*/
    }
    if(!impl->launchedShadeKernel)
    {
        impl->gpuScene.shadowPass.params.origins = impl->shadowQueue.origins;
        impl->gpuScene.shadowPass.params.directions = impl->shadowQueue.directions;
        impl->gpuScene.shadowPass.params.maxDistances = impl->shadowQueue.maxDistances;
        impl->gpuScene.shadowPass.params.transmittance = impl->shadowQueue.transmittance;

        impl->gpuScene.shadowPass.params.activeCount = h_shadowCount;
        cudaMemcpy(reinterpret_cast<void *>(impl->gpuScene.shadowPass.d_params),
                   &impl->gpuScene.shadowPass.params, sizeof(LaunchShadowParams), cudaMemcpyHostToDevice);

        OPTIX_CHECK(optixLaunch(impl->gpuScene.shadowPass.pipeline,
                                0, // stream
                                impl->gpuScene.shadowPass.d_params, sizeof(LaunchShadowParams),
                                &impl->gpuScene.shadowPass.sbt, h_shadowCount, 1, 1));
        accumulateShadowKernel<<<gridForCount(h_shadowCount), block1D>>>(
            impl->shadowQueue.contributions, impl->shadowQueue.transmittance, impl->shadowQueue.pixelIndices,
            impl->d_accumBuffer, h_shadowCount);
    }
    // -------------------------------------------------------------------------
    // 3. Accumulation dans d_accumBuffer
    // -------------------------------------------------------------------------

    dim3 gridAllPixels((pixelCount + block1D.x - 1) / block1D.x);

    // -------------------------------------------------------------------------
    // 4. Incrément du sample count
    // -------------------------------------------------------------------------

    impl->sampleCount++;

    // -------------------------------------------------------------------------
    // 6. Bloom + normalize
    // -------------------------------------------------------------------------

    applyBloom();
    // -------------------------------------------------------------------------
    // 7. Mapping OpenGL / CUDA
    // -------------------------------------------------------------------------
    if(!outputImage)
    {
        finalizeImageNoRender(convergence);
        return impl->value;
    }
    cudaArray_t textureArray;

    cudaGraphicsMapResources(1, &impl->cudaTextureResource, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsMapResources error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }
    cudaGraphicsSubResourceGetMappedArray(&textureArray, impl->cudaTextureResource, 0, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsSubResourceGetMappedArray error: " << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);

        return -1.f;
    }

    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));

    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0;

    cudaCreateSurfaceObject(&surfaceObject, &resourceDesc);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaCreateSurfaceObject error: " << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);

        return -1.f;
    }

    // -------------------------------------------------------------------------
    // 8. Finalisation image
    // -------------------------------------------------------------------------

    finalizeImageV2<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_bloomBuffer,
                                                         impl->d_hdrBloomBuffer, surfaceObject, impl->width,
                                                         impl->height, impl->exposure, impl->bloomStrength);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "finalizeImage error: " << cudaGetErrorString(err) << std::endl;
        cudaDestroySurfaceObject(surfaceObject);
        cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);
        return -1.f;
    }

    cudaDestroySurfaceObject(surfaceObject);

    cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsUnmapResources error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compareBuffers<<<impl->gridSize, impl->blockSize>>>(impl->d_hdrBloomBuffer, impl->d_convergenceBuffer,
                                                            impl->width, impl->height, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergenceBuffer, impl->d_hdrBloomBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
    }

    return impl->value;
}

void Renderer::finalizeImageNoRender(bool convergence)
{
    // -------------------------------------------------------------------------
    // 8. Finalisation image
    // -------------------------------------------------------------------------

    finalizeImageV2NoRender<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_bloomBuffer,
                                                                impl->d_hdrBloomBuffer, impl->width,
                                                                impl->height, impl->exposure, impl->bloomStrength);

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compareBuffers<<<impl->gridSize, impl->blockSize>>>(impl->d_hdrBloomBuffer, impl->d_convergenceBuffer,
                                                            impl->width, impl->height, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergenceBuffer, impl->d_hdrBloomBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
    }
}

float Renderer::render(bool outputImage, bool convergence)
{
    float result = 0.f;
    switch (impl->renderMode)
    {
    case RenderMode::Megakernel:
        result = renderFrame(outputImage, convergence);
        break;
    case RenderMode::Wavefront:
        result = renderFrameWavefront(outputImage, convergence);
        break;
    }
    return result;
}

void Renderer::cleanUp()
{
    impl->currentQueue.destroy();
    impl->nextQueue.destroy();
    impl->sortedQueue.destroy();
    impl->hitBuffers.destroy();
    impl->shadowQueue.destroy();
    cudaFree(impl->d_accumBuffer);
    cudaFree(impl->d_shadowCount);
    cudaFree(impl->d_value);

    cudaFree(impl->d_ranges);

    cudaFree(impl->d_normalizedBuffer);

    cudaFree(impl->d_keys);
    cudaFree(impl->d_values);
    cudaFree(impl->d_octantKeys);

    cudaFree(impl->d_bloomBuffer);
    cudaFree(impl->d_hdrBloomBuffer);
    cudaFree(impl->d_tempBuffer);

    cudaFree(impl->d_convergenceBuffer);

    cudaFree(impl->d_lvl1);
    cudaFree(impl->d_lvl2);

    cudaStreamDestroy(impl->stream);
}

float3 *Renderer::getFinalizedImage()
{
    if (!impl->d_normalizedBuffer)
    {
        std::cerr << "Error: d_normalizedBuffer is null" << std::endl;
        return nullptr;
    }

    // Allocate CPU memory for the image
    float3 *h_image = (float3 *)malloc(impl->hdrBufferSize);

    if (!h_image)
    {
        std::cerr << "Error: Failed to allocate CPU memory for finalized image" << std::endl;
        return nullptr;
    }

    // Copy GPU buffer to CPU
    cudaError_t err = cudaMemcpy(h_image, impl->d_normalizedBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaMemcpy failed - " << cudaGetErrorString(err) << std::endl;
        free(h_image);
        return nullptr;
    }

    return h_image;
}