#include "renderer.hpp"

#include "../../../devicePrograms/launch_params.cuh"
#include "camera/camera.cuh"
#include "scene/scene.cuh"

#include "integrators/pathtracer_integrator.cuh"

#include "../utils/defines.hpp"
#include "utils/constant.cuh"
#include "utils/fill_buffers.cuh"
#include "utils/macro.cuh"

#include "renderingUtils/post_treatment.cuh"
#include "renderingUtils/shading_kernels.cuh"

#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>

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

    float3 *d_throughput = nullptr;
    float3 *d_radiance = nullptr;

    int *d_pixelIndices = nullptr;
    RNG *d_rng;

    uint depth = 0;

    bool *d_isInside = nullptr;
    bool *d_lastBounceWasDelta = nullptr;
    float *d_lastBsdfPdf = nullptr;

    float3 *d_accumBuffer = nullptr;
    float3 *d_normalizedBuffer = nullptr;

    float3 *d_brightBuffer = nullptr;
    float3 *d_bloomBuffer = nullptr;

    float3 *d_tempBuffer = nullptr;
    float3 *d_finalHDRBuffer = nullptr;

    float3 *d_convergenceBuffer = nullptr;

    float3 *d_origins = nullptr;
    float4 *d_directions = nullptr;

    // Hit data in SoA format
    float4 *d_hitPositions = nullptr;
    float4 *d_hitNormals = nullptr;
    int *d_hitMaterialIndices = nullptr;
    
    int *d_hitMask = nullptr;

    int *d_missQueue = nullptr;
    int *d_lambertQueue = nullptr;
    int *d_metalQueue = nullptr;
    int *d_plasticQueue = nullptr;
    int *d_mirrorQueue = nullptr;
    int *d_transparentQueue = nullptr;
    int *d_emissiveQueue = nullptr;

    int *d_missCount = nullptr;
    int *d_lambertCount = nullptr;
    int *d_metalCount = nullptr;
    int *d_plasticCount = nullptr;
    int *d_mirrorCount = nullptr;
    int *d_transparentCount = nullptr;
    int *d_emissiveCount = nullptr;

    int *d_activeQueue = nullptr;
    int *d_nextActiveQueue = nullptr;

    int *d_activeCount = nullptr;
    int *d_nextActiveCount = nullptr;

    dim3 blockSize = dim3(16, 16);

    cudaStream_t stream = nullptr;

    dim3 gridSize;

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
inline dim3 gridForCount(int count, int blockSize = 256) { return dim3((count + blockSize - 1) / blockSize); }

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
    float3 c_sunDirection = toFloat3(sunDir);
    int c_skyColorSamples = 4;
    float c_hr = 1.f / 7994.f;
    float c_hm = 1.f / 1200.f;
    float3 c_betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    float3 c_betaM = make_float3(21e-6f);
    float c_exposure = 1.f;
    float c_sizeAtmosphere = 60000.f;
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
}

void Renderer::init(int p_width, int p_height, float sunDirx, float sunDiry, float sunDirz)
{
    impl->h_camera = initCamera(p_width, p_height);
    initConstant(p_width, p_height, impl->h_camera, make_float4(sunDirx, sunDiry, sunDirz, 0.f));
    impl->width = p_width;
    impl->height = p_height;

    float4 sunDir = make_float4(sunDirx, sunDiry, sunDirz, 0.f);

    setSeed(43);

    // impl->gpuScene = spheresScene(sunDir);
    // impl->gpuScene = implicitSpheresScene(sunDir);
    impl->gpuScene = singleObject(sunDir);
    impl->hdrBufferSize = impl->width * impl->height * sizeof(float3);

    // =========================
    // GPU buffers
    // =========================

    cudaMalloc(&impl->d_throughput, impl->width * impl->height * sizeof(float3));
    cudaMalloc(&impl->d_radiance, impl->width * impl->height * sizeof(float3));
    cudaMalloc(&impl->d_pixelIndices, impl->width * impl->height * sizeof(int));
    cudaMalloc(&impl->d_rng, impl->width * impl->height * sizeof(RNG));
    cudaMalloc(&impl->d_isInside, impl->width * impl->height * sizeof(bool));
    cudaMalloc(&impl->d_lastBounceWasDelta, impl->width * impl->height * sizeof(bool));
    cudaMalloc(&impl->d_lastBsdfPdf, impl->width * impl->height * sizeof(float));

    cudaMalloc(&impl->d_accumBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_normalizedBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_brightBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_bloomBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_tempBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_finalHDRBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_convergenceBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_value, sizeof(float));

    size_t pixelCount = impl->width * impl->height;

    cudaMalloc(&impl->d_missQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_lambertQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_metalQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_plasticQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_mirrorQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_transparentQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_emissiveQueue, pixelCount * sizeof(int));

    cudaMalloc(&impl->d_missCount, sizeof(int));
    cudaMalloc(&impl->d_lambertCount, sizeof(int));
    cudaMalloc(&impl->d_metalCount, sizeof(int));
    cudaMalloc(&impl->d_plasticCount, sizeof(int));
    cudaMalloc(&impl->d_mirrorCount, sizeof(int));
    cudaMalloc(&impl->d_transparentCount, sizeof(int));
    cudaMalloc(&impl->d_emissiveCount, sizeof(int));

    cudaMalloc(&impl->d_origins, impl->hdrBufferSize);
    cudaMalloc(&impl->d_directions, pixelCount * sizeof(float4));
    cudaMalloc(&impl->d_hitPositions, pixelCount * sizeof(float4));
    cudaMalloc(&impl->d_hitNormals, pixelCount * sizeof(float4));
    cudaMalloc(&impl->d_hitMaterialIndices, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_hitMask, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_activeQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_nextActiveQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_activeCount, sizeof(int));
    cudaMalloc(&impl->d_nextActiveCount, sizeof(int));

    // =========================
    // Clear buffers
    // =========================

    cudaMemset(impl->d_accumBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_normalizedBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_brightBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_bloomBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_finalHDRBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_convergenceBuffer, 0, impl->hdrBufferSize);

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
void generatePrimaryRaysKernel(float4 *directions, RNG *p_rng, int *p_pixelIndices, int *activeQueue, int *activeCount,
                               int width, int height, int sampleCount, float invWidth, float invHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = y * width + x;
    p_pixelIndices[pixelIndex] = pixelIndex;

    activeQueue[pixelIndex] = pixelIndex;

    uint seed = (pixelIndex * 0x9E3779B9u) ^ (sampleCount * 0x6C078965u);
    RNG rng(seed);

    float sx = (x + rng.nextFloat()) * invWidth;
    float sy = (y + rng.nextFloat()) * invHeight;
    p_rng[pixelIndex] = rng;
    float3 rayTarget = camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV;
    float3 dir = normalize(rayTarget - camera.cameraPos);
    directions[pixelIndex] = make_float4(dir, 0.f);

    if (pixelIndex == 0)
        *activeCount = width * height;
}

/*
// WAVEFRONT INTERSECTION
GLOBAL
void wavefrontIntersectKernel(CudaScene scene, Ray *rays, OptixHit *hits, int *hitMask,
                              const int *activeQueue, int activeCount, float tMin, float tMax)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    OptixHit hit;
    hitMask[idx] = 0;
    if (scene.intersect(rays[idx], tMin, tMax, hit))
    {
        hits[idx] = hit;
        hitMask[idx] = 1;
    }
}*/

// WAVEFRONT ACCUMULATION
GLOBAL
void accumulateWavefrontKernel(int *pixelIndices, float3 *radiance, float3 *accumBuffer, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stateCount = width * height;

    if (idx >= stateCount)
        return;

    int pixelIndex = pixelIndices[idx];

    if (pixelIndex < 0 || pixelIndex >= width * height)
        return;

    accumBuffer[pixelIndex] += radiance[idx];
}

void Renderer::applyBloom()
{
    cudaMemset(impl->d_brightBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_bloomBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_finalHDRBuffer, 0, impl->hdrBufferSize);

    // =========================
    // Extract bright pixels
    // =========================

    extractBright<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_brightBuffer, impl->width,
                                                       impl->height, impl->threshold);

    // =========================
    // Blur bloom
    // =========================

    applyMultiScaleBloom(impl->d_brightBuffer, impl->d_tempBuffer, impl->d_lvl1, impl->d_lvl2, impl->w1, impl->h1,
                         impl->w2, impl->h2, impl->width, impl->height);

    cudaMemcpy(impl->d_bloomBuffer, impl->d_brightBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
    // =========================
    // Compose final HDR
    // =========================

    addBloom<<<impl->gridSize, impl->blockSize>>>(impl->d_normalizedBuffer, impl->d_bloomBuffer, impl->d_finalHDRBuffer,
                                                  impl->width, impl->height, impl->bloomStrength);
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

    normalizeKernel<<<impl->gridSize, impl->blockSize>>>(impl->d_accumBuffer, impl->d_normalizedBuffer,
                                                         impl->sampleCount, impl->width, impl->height);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "normalizeKernel error: " << cudaGetErrorString(err) << std::endl;
    }

    applyBloom();

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compareBuffers<<<impl->gridSize, impl->blockSize>>>(impl->d_finalHDRBuffer, impl->d_convergenceBuffer,
                                                            impl->width, impl->height, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergenceBuffer, impl->d_finalHDRBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
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

    finalizeImage<<<impl->gridSize, impl->blockSize>>>(impl->d_finalHDRBuffer, surface, impl->width, impl->height,
                                                       impl->exposure);

    cudaDestroySurfaceObject(surface);

    cudaGraphicsUnmapResources(1, &impl->cudaTextureResource);

    return impl->value;
}

GLOBAL
void classifyMaterialKernel(CudaScene scene, const int *activeQueue, int activeCount, const int *hitMask,
                            const int *hitMaterialIndices, int *missQueue, int *missCount, int *lambertQueue, int *lambertCount,
                            int *metalQueue, int *metalCount, int *plasticQueue, int *plasticCount, int *mirrorQueue,
                            int *mirrorCount, int *transparentQueue, int *transparentCount, int *emissiveQueue,
                            int *emissiveCount)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    if (!hitMask[idx])
    {
        int dst = atomicAdd(missCount, 1);
        missQueue[dst] = idx;
        return;
    }

    int materialIndex = hitMaterialIndices[idx];
    const Material &mtl = scene.materials[materialIndex];

    switch (mtl.type())
    {
    case LAMBERT: {
        int dst = atomicAdd(lambertCount, 1);
        lambertQueue[dst] = idx;
        break;
    }

    case METAL: {
        int dst = atomicAdd(metalCount, 1);
        metalQueue[dst] = idx;
        break;
    }

    case PLASTIC: {
        int dst = atomicAdd(plasticCount, 1);
        plasticQueue[dst] = idx;
        break;
    }

    case MIRROR: {
        int dst = atomicAdd(mirrorCount, 1);
        mirrorQueue[dst] = idx;
        break;
    }

    case TRANSPARENT: {
        int dst = atomicAdd(transparentCount, 1);
        transparentQueue[dst] = idx;
        break;
    }

    case EMISSIVE: {
        int dst = atomicAdd(emissiveCount, 1);
        emissiveQueue[dst] = idx;
        break;
    }
    }
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
    cudaMemset(impl->d_activeCount, 0, sizeof(int));
    float invWidth = 1.f / (float)(impl->width - 1);
    float invHeight = 1.f / (float)(impl->height - 1);
    generatePrimaryRaysKernel<<<grid2D, block2D>>>(impl->d_directions, impl->d_rng, impl->d_pixelIndices,
                                                   impl->d_activeQueue, impl->d_activeCount, impl->width, impl->height,
                                                   impl->sampleCount, invWidth, invHeight);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "generatePrimaryRaysKernel error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    int threads = 256;
    int blocks = (pixelCount + threads - 1) / threads;

    initFloat3Buffer<<<blocks, threads>>>(impl->d_origins, pixelCount, impl->h_camera.cameraPos);

    initFloat3Buffer<<<blocks, threads>>>(impl->d_throughput, pixelCount, make_float3(1.f));

    initFloat3Buffer<<<blocks, threads>>>(impl->d_radiance, pixelCount, make_float3(0.f));

    initBoolBuffer<<<blocks, threads>>>(impl->d_isInside, pixelCount, false);

    initBoolBuffer<<<blocks, threads>>>(impl->d_lastBounceWasDelta, pixelCount, false);

    initFloatBuffer<<<blocks, threads>>>(impl->d_lastBsdfPdf, pixelCount, 1.f);

    int h_activeCount = pixelCount;

    // -------------------------------------------------------------------------
    // 2. Boucle wavefront activeQueue
    // -------------------------------------------------------------------------

    for (int bounce = 0; bounce < impl->maxBounces; bounce++)
    {

        if (h_activeCount == 0)
            break;

        cudaMemset(impl->d_nextActiveCount, 0, sizeof(int));

        cudaMemset(impl->d_missCount, 0, sizeof(int));

        cudaMemset(impl->d_lambertCount, 0, sizeof(int));
        cudaMemset(impl->d_metalCount, 0, sizeof(int));
        cudaMemset(impl->d_plasticCount, 0, sizeof(int));
        cudaMemset(impl->d_mirrorCount, 0, sizeof(int));
        cudaMemset(impl->d_transparentCount, 0, sizeof(int));
        cudaMemset(impl->d_emissiveCount, 0, sizeof(int));
        // ---------------------------------------------------------------------
        // 2.1 Intersection uniquement des rayons actifs
        // ---------------------------------------------------------------------
        impl->gpuScene.optixData.launchParams.origins = impl->d_origins;
        impl->gpuScene.optixData.launchParams.directions = impl->d_directions;
        impl->gpuScene.optixData.launchParams.hitPositions = impl->d_hitPositions;
        impl->gpuScene.optixData.launchParams.hitNormals = impl->d_hitNormals;
        impl->gpuScene.optixData.launchParams.hitMaterialIndices = impl->d_hitMaterialIndices;
        impl->gpuScene.optixData.launchParams.hitMask = impl->d_hitMask;
        impl->gpuScene.optixData.launchParams.activeQueue = impl->d_activeQueue;
        impl->gpuScene.optixData.launchParams.activeCount = h_activeCount;
        cudaMemcpy(reinterpret_cast<void *>(impl->gpuScene.optixData.d_launchParams),
                   &impl->gpuScene.optixData.launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice);

        OPTIX_CHECK(optixLaunch(impl->gpuScene.optixData.pipeline,
                                0, // stream
                                impl->gpuScene.optixData.d_launchParams, sizeof(LaunchParams),
                                &impl->gpuScene.optixData.sbt, h_activeCount, 1, 1));
                                
        // ---------------------------------------------------------------------
        // 2.2 Shading + compaction nextActiveQueue
        // ---------------------------------------------------------------------
        classifyMaterialKernel<<<gridForCount(h_activeCount), block1D>>>(
            impl->gpuScene, impl->d_activeQueue, h_activeCount, impl->d_hitMask, impl->d_hitMaterialIndices, impl->d_missQueue,
            impl->d_missCount, impl->d_lambertQueue, impl->d_lambertCount, impl->d_metalQueue, impl->d_metalCount,
            impl->d_plasticQueue, impl->d_plasticCount, impl->d_mirrorQueue, impl->d_mirrorCount,
            impl->d_transparentQueue, impl->d_transparentCount, impl->d_emissiveQueue, impl->d_emissiveCount);

        int h_missCount = 0;
        int h_lambertCount = 0;
        int h_metalCount = 0;
        int h_plasticCount = 0;
        int h_mirrorCount = 0;
        int h_transparentCount = 0;
        int h_emissiveCount = 0;

        cudaMemcpy(&h_missCount, impl->d_missCount, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(&h_lambertCount, impl->d_lambertCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_metalCount, impl->d_metalCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_plasticCount, impl->d_plasticCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_mirrorCount, impl->d_mirrorCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_transparentCount, impl->d_transparentCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_emissiveCount, impl->d_emissiveCount, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_missCount > 0)
        {
            shadeMissKernel<<<gridForCount(h_missCount), block1D>>>(impl->d_origins, impl->d_directions,
                                                                    impl->d_throughput, impl->d_radiance,
                                                                    impl->d_missQueue, h_missCount, bounce == 0);
        }
        if (h_lambertCount > 0)
        {
            shadeLambertKernel<<<gridForCount(h_lambertCount), block1D>>>(
                impl->gpuScene, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_radiance, impl->d_rng,
                impl->d_hitPositions, impl->d_hitNormals, impl->d_hitMaterialIndices,
                impl->d_lastBounceWasDelta, impl->d_lambertQueue, h_lambertCount, impl->d_nextActiveQueue, impl->d_nextActiveCount,
                bounce);
        }
        if (h_metalCount > 0)
        {
            shadeMetalKernel<<<gridForCount(h_metalCount), block1D>>>(
                impl->gpuScene, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_radiance, impl->d_rng,
                impl->d_hitPositions, impl->d_hitNormals, impl->d_hitMaterialIndices,
                impl->d_lastBounceWasDelta, impl->d_metalQueue, h_metalCount, impl->d_nextActiveQueue, impl->d_nextActiveCount,
                bounce);
        }
        if (h_plasticCount > 0)
        {
            shadePlasticNEEKernel<<<gridForCount(h_plasticCount), block1D>>>(
                impl->gpuScene, impl->d_directions, impl->d_throughput, impl->d_radiance, impl->d_rng,
                impl->d_hitPositions, impl->d_hitNormals, impl->d_hitMaterialIndices,
                impl->d_plasticQueue, h_plasticCount, impl->gpuScene.nbLights);

            shadePlasticKernel<<<gridForCount(h_plasticCount), block1D>>>(
                impl->gpuScene.materials, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_rng,
                impl->d_hitPositions, impl->d_hitNormals, impl->d_hitMaterialIndices,
                impl->d_lastBounceWasDelta, impl->d_plasticQueue, h_plasticCount, impl->d_nextActiveQueue, impl->d_nextActiveCount,
                bounce);
        }
        if (h_mirrorCount > 0)
        {
            shadeMirrorKernel<<<gridForCount(h_mirrorCount), block1D>>>(
                impl->gpuScene, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_rng,
                impl->d_lastBounceWasDelta, impl->d_lastBsdfPdf, impl->d_hitPositions, impl->d_hitNormals,
                impl->d_hitMaterialIndices, impl->d_mirrorQueue, h_mirrorCount, impl->d_nextActiveQueue, impl->d_nextActiveCount, bounce);
        }
        if (h_transparentCount > 0)
        {
            shadeTransparentKernel<<<gridForCount(h_transparentCount), block1D>>>(
                impl->gpuScene, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_rng, impl->d_isInside,
                impl->d_lastBounceWasDelta, impl->d_lastBsdfPdf, impl->d_hitPositions, impl->d_hitNormals,
                impl->d_hitMaterialIndices, impl->d_transparentQueue, h_transparentCount, impl->d_nextActiveQueue, impl->d_nextActiveCount, bounce);
        }
        if (h_emissiveCount > 0)
        {
            shadeEmissiveKernel<<<gridForCount(h_emissiveCount), block1D>>>(
                impl->gpuScene, impl->d_origins, impl->d_directions, impl->d_throughput, impl->d_radiance,
                impl->d_lastBounceWasDelta, impl->d_lastBsdfPdf, impl->d_hitMaterialIndices,
                impl->d_hitMask, impl->d_emissiveQueue, h_emissiveCount);
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

        cudaMemcpy(&h_activeCount, impl->d_nextActiveCount, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap activeQueue / nextActiveQueue
        std::swap(impl->d_activeQueue, impl->d_nextActiveQueue);

        std::swap(impl->d_activeCount, impl->d_nextActiveCount);
    }

    // -------------------------------------------------------------------------
    // 3. Accumulation dans d_accumBuffer
    // -------------------------------------------------------------------------

    dim3 gridAllPixels((pixelCount + block1D.x - 1) / block1D.x);

    accumulateWavefrontKernel<<<gridAllPixels, block1D>>>(impl->d_pixelIndices, impl->d_radiance, impl->d_accumBuffer,
                                                          impl->width, impl->height);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "accumulateWavefrontKernel error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    // -------------------------------------------------------------------------
    // 4. Incrément du sample count
    // -------------------------------------------------------------------------

    impl->sampleCount++;

    // -------------------------------------------------------------------------
    // 5. Normalisation HDR
    // -------------------------------------------------------------------------

    normalizeKernel<<<impl->gridSize, impl->blockSize>>>(impl->d_accumBuffer, impl->d_normalizedBuffer,
                                                         impl->sampleCount, impl->width, impl->height);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "normalizeKernel error: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    // -------------------------------------------------------------------------
    // 6. Bloom
    // -------------------------------------------------------------------------

    applyBloom();

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compareBuffers<<<impl->gridSize, impl->blockSize>>>(impl->d_finalHDRBuffer, impl->d_convergenceBuffer,
                                                            impl->width, impl->height, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergenceBuffer, impl->d_finalHDRBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToDevice);
    }

    if (!outputImage)
        return impl->value;
    // -------------------------------------------------------------------------
    // 7. Mapping OpenGL / CUDA
    // -------------------------------------------------------------------------

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

    finalizeImage<<<impl->gridSize, impl->blockSize>>>(impl->d_finalHDRBuffer, surfaceObject, impl->width, impl->height,
                                                       impl->exposure);

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

    return impl->value;
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
    cudaFree(impl->d_value);
    cudaFree(impl->d_accumBuffer);

    cudaFree(impl->d_throughput);
    cudaFree(impl->d_radiance);
    cudaFree(impl->d_pixelIndices);
    cudaFree(impl->d_rng);
    cudaFree(impl->d_isInside);
    cudaFree(impl->d_lastBounceWasDelta);
    cudaFree(impl->d_lastBsdfPdf);
    cudaFree(impl->d_normalizedBuffer);

    cudaFree(impl->d_missQueue);
    cudaFree(impl->d_lambertQueue);
    cudaFree(impl->d_metalQueue);
    cudaFree(impl->d_plasticQueue);
    cudaFree(impl->d_mirrorQueue);
    cudaFree(impl->d_transparentQueue);
    cudaFree(impl->d_emissiveQueue);

    cudaFree(impl->d_missCount);
    cudaFree(impl->d_lambertCount);
    cudaFree(impl->d_metalCount);
    cudaFree(impl->d_plasticCount);
    cudaFree(impl->d_mirrorCount);
    cudaFree(impl->d_transparentCount);
    cudaFree(impl->d_emissiveCount);

    cudaFree(impl->d_brightBuffer);

    cudaFree(impl->d_bloomBuffer);

    cudaFree(impl->d_tempBuffer);

    cudaFree(impl->d_finalHDRBuffer);

    cudaFree(impl->d_convergenceBuffer);

    cudaFree(impl->d_origins);
    cudaFree(impl->d_directions);
    cudaFree(impl->d_hitPositions);
    cudaFree(impl->d_hitNormals);
    cudaFree(impl->d_hitMaterialIndices);
    cudaFree(impl->d_hitMask);
    cudaFree(impl->d_activeQueue);
    cudaFree(impl->d_activeCount);
    cudaFree(impl->d_nextActiveQueue);
    cudaFree(impl->d_nextActiveCount);

    cudaFree(impl->d_lvl1);

    cudaFree(impl->d_lvl2);

    cudaStreamDestroy(impl->stream);
}

float3 *Renderer::getFinalizedImage()
{
    if (!impl->d_finalHDRBuffer)
    {
        std::cerr << "Error: d_finalHDRBuffer is null" << std::endl;
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
    cudaError_t err = cudaMemcpy(h_image, impl->d_finalHDRBuffer, impl->hdrBufferSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaMemcpy failed - " << cudaGetErrorString(err) << std::endl;
        free(h_image);
        return nullptr;
    }

    // Apply finalization processing on CPU (same as finalizeImage kernel)
    int pixelCount = impl->width * impl->height;
    for (int i = 0; i < pixelCount; i++)
    {
        float3 c = h_image[i];

        // Reinhard tonemap
        c = (c * impl->exposure) / (make_float3(1.f) + c * impl->exposure);

        // Gamma correction
        c = make_float3(sqrtf(fmaxf(c.x, 0.f)), sqrtf(fmaxf(c.y, 0.f)), sqrtf(fmaxf(c.z, 0.f)));

        // Clamp to [0, 1]
        c.x = fminf(c.x, 1.f);
        c.y = fminf(c.y, 1.f);
        c.z = fminf(c.z, 1.f);

        h_image[i] = c;
    }

    return h_image;
}