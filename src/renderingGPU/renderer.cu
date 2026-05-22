#include "renderer.hpp"

#include "camera/camera.cuh"
#include "scene/scene.cuh"

#include "integrators/pathtracer_integrator.cuh"

#include "../utils/defines.hpp"
#include "utils/constant.cuh"
#include "utils/macro.cuh"

#include "renderingUtils/post_treatment.cuh"
#include "renderingUtils/shading_kernels.cuh"
#include "renderingUtils/wavefront_state.cuh"

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
    RenderMode renderMode = RenderMode::Megakernel;
    int width = 1920;
    int height = 1080;

    float threshold = 1.0f;
    float bloomStrength = 0.25f;
    float exposure = 1.0f;

    int sampleCount = 0;
    int maxBounces = 5;
    size_t hdrBufferSize = 0;

    CudaScene gpuScene;

    float3 *d_accumBuffer = nullptr;
    float3 *d_normalizedBuffer = nullptr;

    float3 *d_brightBuffer = nullptr;
    float3 *d_bloomBuffer = nullptr;

    float3 *d_tempBuffer = nullptr;
    float3 *d_finalHDRBuffer = nullptr;

    WavefrontState *d_wavefrontStates = nullptr;

    HitRecord *d_hits = nullptr;
    int *d_hitMask = nullptr;

    int *d_activeQueue = nullptr;
    int *d_nextActiveQueue = nullptr;

    int *d_activeCount = nullptr;
    int *d_nextActiveCount = nullptr;

    unsigned char *d_finalBuffer = nullptr;

    dim3 blockSize = dim3(16, 16);

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
    Camera camera = Camera{
        fov, aspect, focalDistance, toFloat4(camPos), toFloat4(topLeft), toFloat4(viewportU), toFloat4(viewportV)};
    return camera;
}

HOST void initConstant(int width, int height, float4 sunDir)
{
    int c_nbBounces = 8;
    float c_earthRadius = 6360e3f;
    float3 c_sunDirection = toFloat3(sunDir);
    int c_skyColorSamples = 4;
    float c_hr = 7994.f;
    float c_hm = 1200.f;
    float3 c_betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    float3 c_betaM = make_float3(21e-6f);
    float c_exposure = 1.f;
    Camera c_camera = initCamera(width, height);
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
    initConstant(p_width, p_height, make_float4(sunDirx, sunDiry, sunDirz, 0.f));
    impl->width = p_width;
    impl->height = p_height;

    float4 sunDir = make_float4(sunDirx, sunDiry, sunDirz, 0.f);

    setSeed(43);

    impl->gpuScene = spheresScene(sunDir);
    // impl->gpuScene = implicitSpheresScene(sunDir);
    // impl->gpuScene = singleObject(sunDir);
    impl->hdrBufferSize = impl->width * impl->height * sizeof(float3);

    size_t finalBufferSize = impl->width * impl->height * 3 * sizeof(unsigned char);

    // =========================
    // GPU buffers
    // =========================

    cudaMalloc(&impl->d_accumBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_normalizedBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_brightBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_bloomBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_tempBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_finalHDRBuffer, impl->hdrBufferSize);

    size_t pixelCount = impl->width * impl->height;
    cudaMalloc(&impl->d_wavefrontStates, pixelCount * sizeof(WavefrontState));
    cudaMalloc(&impl->d_hits, pixelCount * sizeof(HitRecord));
    cudaMalloc(&impl->d_hitMask, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_activeQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_nextActiveQueue, pixelCount * sizeof(int));
    cudaMalloc(&impl->d_activeCount, sizeof(int));
    cudaMalloc(&impl->d_nextActiveCount, sizeof(int));

    cudaMalloc(&impl->d_finalBuffer, finalBufferSize);

    // =========================
    // Clear buffers
    // =========================

    cudaMemset(impl->d_accumBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_normalizedBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_brightBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_bloomBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_finalHDRBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_finalBuffer, 0, finalBufferSize);

    // =========================
    // RNG
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

// MEGAKERNEL
GLOBAL
void renderKernel(CudaScene gpuScene, float3 *d_accumBuffer, int width, int height, int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = y * width + x;
    uint seed = pixelIndex ^ (sampleCount * 0x9E3779B9u);
    RNG localState(seed);

    float3 finalColor = make_float3(0.f);

    float sx = (x + localState.nextFloat()) / (float)(width - 1);
    float sy = (y + localState.nextFloat()) / (float)(height - 1);

    float3 rayTarget = toFloat3(camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV);
    float3 direction = normalize(rayTarget - toFloat3(camera.cameraPos));

    Ray ray(toFloat3(camera.cameraPos), direction);
    finalColor += PathtracerIntegrator::lighting(gpuScene, ray, 0, 1e20f, &localState);

    d_accumBuffer[pixelIndex] += finalColor;
}

// WAVEFRONT INIT
GLOBAL
void generatePrimaryRaysKernel(WavefrontState *states, int *activeQueue, int *activeCount, int width, int height,
                               int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = y * width + x;

    uint seed = pixelIndex ^ (sampleCount * 0x9E3779B9u);
    RNG rng(seed);

    float sx = (x + rng.nextFloat()) / (float)(width - 1);
    float sy = (y + rng.nextFloat()) / (float)(height - 1);

    float3 rayTarget = toFloat3(camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV);

    float3 direction = normalize(rayTarget - toFloat3(camera.cameraPos));

    Ray primaryRay(toFloat3(camera.cameraPos), direction);

    states[pixelIndex] = WavefrontState(primaryRay, rng, pixelIndex);

    activeQueue[pixelIndex] = pixelIndex;

    if (pixelIndex == 0)
        *activeCount = width * height;
}

// WAVEFRONT INTERSECTION
GLOBAL
void wavefrontIntersectKernel(CudaScene scene, WavefrontState *states, HitRecord *hits, int *hitMask,
                                    const int *activeQueue, int activeCount, float tMin, float tMax)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    if (qid >= activeCount)
        return;

    int idx = activeQueue[qid];

    WavefrontState state = states[idx];

    if (!state.active)
    {
        hitMask[idx] = 0;
        return;
    }

    HitRecord hit;

    if (scene.intersect(state.ray, tMin, tMax, hit))
    {
        hits[idx] = hit;
        hitMask[idx] = 1;
    }
    else
    {
        hitMask[idx] = 0;
    }
}

// WAVEFRONT ACCUMULATION
GLOBAL
void accumulateWavefrontKernel(const WavefrontState *wavefrontStates, float3 *accumBuffer, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stateCount = width * height;

    if (idx >= stateCount)
        return;

    const WavefrontState &state = wavefrontStates[idx];

    int pixelIndex = state.pixelIndex;

    if (pixelIndex < 0 || pixelIndex >= width * height)
        return;

    accumBuffer[pixelIndex] += state.radiance;
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

void Renderer::renderFrame(bool outputImage)
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

    if (!outputImage)
        return;

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
}

void Renderer::renderFrameWavefront(bool outputImage)
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

    generatePrimaryRaysKernel<<<grid2D, block2D>>>(impl->d_wavefrontStates, impl->d_activeQueue, impl->d_activeCount,
                                                   impl->width, impl->height, impl->sampleCount);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "generatePrimaryRaysKernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    int h_activeCount = pixelCount;

    // -------------------------------------------------------------------------
    // 2. Boucle wavefront activeQueue
    // -------------------------------------------------------------------------

    for (int bounce = 0; bounce < impl->maxBounces; bounce++)
    {
        if (h_activeCount == 0)
            break;

        cudaMemset(impl->d_nextActiveCount, 0, sizeof(int));

        // ---------------------------------------------------------------------
        // 2.1 Intersection uniquement des rayons actifs
        // ---------------------------------------------------------------------

        wavefrontIntersectKernel<<<gridForCount(h_activeCount), block1D>>>(
            impl->gpuScene, impl->d_wavefrontStates, impl->d_hits, impl->d_hitMask, impl->d_activeQueue, h_activeCount,
            0.001f, 1e20f);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "wavefrontIntersectKernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // ---------------------------------------------------------------------
        // 2.2 Shading + compaction nextActiveQueue
        // ---------------------------------------------------------------------

        shadeWavefrontKernel<<<gridForCount(h_activeCount), block1D>>>(
            impl->gpuScene, impl->d_wavefrontStates, impl->d_hits, impl->d_hitMask, impl->d_activeQueue, h_activeCount,
            impl->d_nextActiveQueue, impl->d_nextActiveCount);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "shadeWavefrontKernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // ---------------------------------------------------------------------
        // 2.3 Récupération du nombre de rayons actifs pour le prochain bounce
        // ---------------------------------------------------------------------

        cudaMemcpy(&h_activeCount, impl->d_nextActiveCount, sizeof(int), cudaMemcpyDeviceToHost);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "cudaMemcpy nextActiveCount error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Swap activeQueue / nextActiveQueue
        std::swap(impl->d_activeQueue, impl->d_nextActiveQueue);

        std::swap(impl->d_activeCount, impl->d_nextActiveCount);
    }

    // -------------------------------------------------------------------------
    // 3. Accumulation dans d_accumBuffer
    // -------------------------------------------------------------------------

    dim3 gridAllPixels((pixelCount + block1D.x - 1) / block1D.x);

    accumulateWavefrontKernel<<<gridAllPixels, block1D>>>(impl->d_wavefrontStates, impl->d_accumBuffer, impl->width,
                                                          impl->height);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "accumulateWavefrontKernel error: " << cudaGetErrorString(err) << std::endl;
        return;
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
        return;
    }

    // -------------------------------------------------------------------------
    // 6. Bloom
    // -------------------------------------------------------------------------

    applyBloom();

    if (!outputImage)
        return;
    // -------------------------------------------------------------------------
    // 7. Mapping OpenGL / CUDA
    // -------------------------------------------------------------------------

    cudaArray_t textureArray;

    cudaGraphicsMapResources(1, &impl->cudaTextureResource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsMapResources error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaGraphicsSubResourceGetMappedArray(&textureArray, impl->cudaTextureResource, 0, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsSubResourceGetMappedArray error: " << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);

        return;
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

        return;
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
    }

    cudaDestroySurfaceObject(surfaceObject);

    cudaGraphicsUnmapResources(1, &impl->cudaTextureResource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsUnmapResources error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

void Renderer::render(bool outputImage)
{
    switch (impl->renderMode)
    {
    case RenderMode::Megakernel:
        renderFrame(outputImage);
        break;
    case RenderMode::Wavefront:
        renderFrameWavefront(outputImage);
        break;
    }
}

void Renderer::cleanUp()
{
    cudaFree(impl->d_accumBuffer);

    cudaFree(impl->d_normalizedBuffer);

    cudaFree(impl->d_brightBuffer);

    cudaFree(impl->d_bloomBuffer);

    cudaFree(impl->d_tempBuffer);

    cudaFree(impl->d_finalHDRBuffer);

    cudaFree(impl->d_wavefrontStates);
    cudaFree(impl->d_hits);
    cudaFree(impl->d_hitMask);
    cudaFree(impl->d_activeQueue);
    cudaFree(impl->d_activeCount);
    cudaFree(impl->d_nextActiveQueue);
    cudaFree(impl->d_nextActiveCount);

    cudaFree(impl->d_finalBuffer);

    cudaFree(impl->d_lvl1);

    cudaFree(impl->d_lvl2);
}