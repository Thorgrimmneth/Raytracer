#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include "scene.cuh"
#include "integrators/whitted_integrator.cuh"

#include "camera/camera.cuh"
#include "../defines.hpp"
#include "hello_cuda.hpp"

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

class Renderer::Impl
{
public:

    int width  = 1920;
    int height = 1080;

    float threshold = 1.0f;
    float bloomStrength = 0.25f;
    float exposure = 1.0f;

    int sampleCount = 0;

    size_t hdrBufferSize = 0;

    CudaScene gpuScene;

    float3* d_accumBuffer = nullptr;
    float3* d_normalizedBuffer = nullptr;

    float3* d_brightBuffer = nullptr;
    float3* d_bloomBuffer = nullptr;

    float3* d_tempBuffer = nullptr;
    float3* d_finalHDRBuffer = nullptr;

    unsigned char* d_finalBuffer = nullptr;

    dim3 blockSize = dim3(16, 16);

    dim3 gridSize;

    int w1 = 0;
    int h1 = 0;

    int w2 = 0;
    int h2 = 0;

    float3* d_lvl1 = nullptr;

    float3* d_lvl2 = nullptr;

    cudaGraphicsResource* cudaTextureResource = nullptr;
};

Renderer::Renderer()
{
    impl = new Impl();
}

Renderer::~Renderer()
{
    cleanup();
    delete impl;
}

void Renderer::setInteropResource(cudaGraphicsResource* resource)
{
    impl->cudaTextureResource = resource;
}

__host__
Camera initCamera(int width, int height){
    // ===== Camera =====
    float3 camPos    = make_float3(8.f, 2.f, 3.f);
    float3 camTarget = make_float3(0.f, 0.f, 0.f);
    float3 camUp     = make_float3(0.f, 1.f, 0.f);

    float fov    = 60.f;
    float aspect = (float)width / (float)height;
    float focalDistance = 1.f;

    // === Base vectors EXACTEMENT comme CPU ===
    float3 w = normalize(camPos - camTarget);
    float3 u = normalize(cross(camUp, w));
    float3 v = normalize(cross(w, u));

    // === Viewport ===
    float theta = fov * 3.14159265f / 180.f;
    float viewportHeight = 2.f * tanf(theta * 0.5f) * focalDistance;
    float viewportWidth  = viewportHeight * aspect;

    float3 viewportU = u * viewportWidth;
    float3 viewportV = v * viewportHeight;

    float3 topLeft =
        camPos
        - w * focalDistance
        + viewportV * 0.5f
        - viewportU * 0.5f;
    Camera camera = Camera{fov, aspect, focalDistance, toFloat4(camPos), toFloat4(topLeft), toFloat4(viewportU), toFloat4(viewportV)};
    return camera;
}

__host__
void initConstant(int width, int height, float4 sunDir){
    int c_nbBounces = 5;
    float c_earthRadius = 6360e3f;
    float3 c_sunDirection = toFloat3(sunDir);
    int c_skyColorSamples = 8;
    float c_hr = 7994.f;
    float c_hm = 1200.f;
    float3 c_betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    float3 c_betaM = make_float3(21e-6f);
    float c_exposure = 1.f;
    Camera c_camera = initCamera(width, height);
    float c_sizeAtmosphere = 60000.f;
    cudaMemcpyToSymbol(nbBounces, &c_nbBounces, sizeof(int));
    cudaMemcpyToSymbol(earthRadius,&c_earthRadius,  sizeof(float));
    cudaMemcpyToSymbol(sunDirection,&c_sunDirection,  sizeof(float3));
    cudaMemcpyToSymbol(skyColorSamples,&c_skyColorSamples,  sizeof(int));
    cudaMemcpyToSymbol(hr,&c_hr,  sizeof(float));
    cudaMemcpyToSymbol(hm,&c_hm,  sizeof(float));
    cudaMemcpyToSymbol(betaR,&c_betaR,  sizeof(float3));
    cudaMemcpyToSymbol(betaM,&c_betaM,  sizeof(float3));
    cudaMemcpyToSymbol(exposure,&c_exposure,  sizeof(float));
    cudaMemcpyToSymbol(camera,&c_camera,  sizeof(Camera));
    cudaMemcpyToSymbol(sizeAtmosphere, &c_sizeAtmosphere, sizeof(float));
}

__global__
void initRNG(curandState* states, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int id = y * width + x;

    curand_init(1337, id, 0, &states[id]);
}

__global__
void renderKernel(
    CudaScene gpuScene,
    float3* d_accumBuffer,
    int width,
    int height,
    int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIndex = y * width + x;
    uint seed = pixelIndex ^ (sampleCount * 0x9E3779B9u);
    RNG localState(seed);

    float3 finalColor = make_float3(0.f);

    float sx = (x + localState.nextFloat()) / (float)(width  - 1);
    float sy = (y + localState.nextFloat()) / (float)(height - 1);

    float3 rayTarget = toFloat3(camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV);
    float3 direction = normalize(rayTarget - toFloat3(camera.cameraPos));

    Ray ray(toFloat3(camera.cameraPos), direction);
    finalColor += WhittedIntegrator::lighting(gpuScene, ray, 0, 1e20f, &localState);

    d_accumBuffer[pixelIndex] += finalColor;
}

__global__
void normalizeKernel(
    float3* accum,
    float3* normalized,
    int sampleCount,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    normalized[idx] =accum[idx] / (float)sampleCount;
}
       

__global__
void extractBright(float3* hdr,
                   float3* bright,
                   int width,
                   int height,
                   float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float3 c = hdr[idx];
    float maxChannel = fmaxf(c.x, fmaxf(c.y, c.z));
    bright[idx] = (maxChannel > threshold) ? c : make_float3(0.f);
}

__global__
void downsample(float3* input,
                float3* output,
                int width,
                int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int newWidth  = width / 2;
    int newHeight = height / 2;

    if (x >= newWidth || y >= newHeight) return;

    int baseX = x * 2;
    int baseY = y * 2;

    int idx00 = baseY * width + baseX;
    int idx10 = baseY * width + baseX + 1;
    int idx01 = (baseY + 1) * width + baseX;
    int idx11 = (baseY + 1) * width + baseX + 1;

    output[y * newWidth + x] =
        (input[idx00] +
         input[idx10] +
         input[idx01] +
         input[idx11]) * 0.25f;
}

__global__
void blurHorizontal(float3* input,
                    float3* output,
                    int width,
                    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    int idx = y * width + x;
    float3 result = input[idx] * weights[0];

    for (int i = 1; i < 5; i++)
    {
        int left  = y * width + max(x - i, 0);
        int right = y * width + min(x + i, width - 1);

        result += input[left]  * weights[i];
        result += input[right] * weights[i];
    }

    output[idx] = result;
}

__global__
void blurVertical(float3* input,
                  float3* output,
                  int width,
                  int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float weights[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    int idx = y * width + x;
    float3 result = input[idx] * weights[0];

    for (int i = 1; i < 5; i++)
    {
        int down = max(y - i, 0) * width + x;
        int up   = min(y + i, height - 1) * width + x;

        result += input[down] * weights[i];
        result += input[up]   * weights[i];
    }

    output[idx] = result;
}

__global__
void upsampleAdd(float3* lowRes,
                 float3* highRes,
                 int lowWidth,
                 int lowHeight,
                 int highWidth,
                 float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= highWidth || y >= lowHeight * 2) return;

    float gx = (x + 0.5f) * 0.5f - 0.5f;
    float gy = (y + 0.5f) * 0.5f - 0.5f;

    int x0 = floorf(gx);
    int y0 = floorf(gy);
    int x1 = min(x0 + 1, lowWidth - 1);
    int y1 = min(y0 + 1, lowHeight - 1);

    float tx = gx - x0;
    float ty = gy - y0;

    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float3 c00 = lowRes[y0 * lowWidth + x0];
    float3 c10 = lowRes[y0 * lowWidth + x1];
    float3 c01 = lowRes[y1 * lowWidth + x0];
    float3 c11 = lowRes[y1 * lowWidth + x1];

    float3 c =
        lerp(lerp(c00, c10, tx),
            lerp(c01, c11, tx),
            ty);

    highRes[y * highWidth + x] += c * strength;
}

void applyMultiScaleBloom(float3* d_bright,
                          float3* d_temp,
                          float3* d_lvl1, float3* d_lvl2,
                          int w1, int h1, int w2, int h2,
                          int width,
                          int height)
{
    dim3 block(16,16);

    dim3 grid1((w1+15)/16,(h1+15)/16);

    downsample<<<grid1,block>>>(d_bright,d_lvl1,width,height);

    blurHorizontal<<<grid1,block>>>(d_lvl1,d_temp,w1,h1);
    blurVertical<<<grid1,block>>>(d_temp,d_lvl1,w1,h1);

    dim3 grid2((w2+15)/16,(h2+15)/16);

    downsample<<<grid2,block>>>(d_lvl1,d_lvl2,w1,h1);

    blurHorizontal<<<grid2,block>>>(d_lvl2,d_temp,w2,h2);
    blurVertical<<<grid2,block>>>(d_temp,d_lvl2,w2,h2);

    upsampleAdd<<<grid1,block>>>(d_lvl2,d_lvl1,w2,h2,w1,1.0f);

    upsampleAdd<<<dim3((width+15)/16,(height+15)/16),block>>>(
        d_lvl1,d_bright,w1,h1,width,1.0f);
}

__global__
void addBloom(float3* hdr,
              float3* bloom,
              float3* out,
              int width,
              int height,
              float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    out[idx] = hdr[idx] + bloom[idx] * strength;
}

__global__
void finalizeImage(
    float3* hdr,
    cudaSurfaceObject_t surface,
    int width,
    int height,
    float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float3 c = hdr[idx];

    // Reinhard tonemap
    c = (c * exposure) / (make_float3(1.f) + c * exposure);

    // Gamma correction
    c = make_float3(
        sqrtf(fmaxf(c.x, 0.f)),
        sqrtf(fmaxf(c.y, 0.f)),
        sqrtf(fmaxf(c.z, 0.f))
    );

    uchar4 pixel = make_uchar4(
        (unsigned char)(255.f * fminf(c.x, 1.f)),
        (unsigned char)(255.f * fminf(c.y, 1.f)),
        (unsigned char)(255.f * fminf(c.z, 1.f)),
        255
    );

    surf2Dwrite(
        pixel,
        surface,
        x * sizeof(uchar4),
        y
    );
}

void Renderer::resetAccumulation()
{
    impl->sampleCount = 0;

    cudaMemset(
        impl->d_accumBuffer,
        0,
        impl->hdrBufferSize
    );
}

void Renderer::init(
    int p_width,
    int p_height,
    float sunDirx,
    float sunDiry,
    float sunDirz)
{
    initConstant(p_width, p_height, make_float4(sunDirx, sunDiry, sunDirz, 0.f));
    impl->width = p_width;
    impl->height = p_height;

    float4 sunDir =
        make_float4(sunDirx, sunDiry, sunDirz, 0.f);

    RT::setSeed(42);

    impl->gpuScene = spheresScene(sunDir);

    impl->hdrBufferSize =
        impl->width * impl->height * sizeof(float3);

    size_t finalBufferSize =
        impl->width * impl->height * 3 * sizeof(unsigned char);

    // =========================
    // GPU buffers
    // =========================

    cudaMalloc(&impl->d_accumBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_normalizedBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_brightBuffer, impl->hdrBufferSize);

    cudaMalloc(&impl->d_bloomBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_tempBuffer, impl->hdrBufferSize);
    cudaMalloc(&impl->d_finalHDRBuffer, impl->hdrBufferSize);

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

    impl->gridSize = dim3(
        (impl->width + impl->blockSize.x - 1) / impl->blockSize.x,
        (impl->height + impl->blockSize.y - 1) / impl->blockSize.y
    );

    // =========================
    // Bloom mip chain
    // =========================

    impl->w1 = impl->width / 2;
    impl->h1 = impl->height / 2;

    impl->w2 = impl->w1 / 2;
    impl->h2 = impl->h1 / 2;

    cudaMalloc(
        &impl->d_lvl1,
        impl->w1 * impl->h1 * sizeof(float3)
    );

    cudaMalloc(
        &impl->d_lvl2,
        impl->w2 * impl->h2 * sizeof(float3)
    );

    impl->sampleCount = 0;
}

void Renderer::applyBloom()
{
    cudaMemset(impl->d_brightBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_bloomBuffer, 0, impl->hdrBufferSize);

    cudaMemset(impl->d_finalHDRBuffer, 0, impl->hdrBufferSize);

    // =========================
    // Extract bright pixels
    // =========================

    extractBright<<<impl->gridSize, impl->blockSize>>>(
        impl->d_normalizedBuffer,
        impl->d_brightBuffer,
        impl->width,
        impl->height,
        impl->threshold
    );

    // =========================
    // Blur bloom
    // =========================

    applyMultiScaleBloom(
        impl->d_brightBuffer,
        impl->d_tempBuffer,
        impl->d_lvl1,
        impl->d_lvl2,
        impl->w1,
        impl->h1,
        impl->w2,
        impl->h2,
        impl->width,
        impl->height
    );

    cudaMemcpy(
        impl->d_bloomBuffer,
        impl->d_brightBuffer,
        impl->hdrBufferSize,
        cudaMemcpyDeviceToDevice
    );
    // =========================
    // Compose final HDR
    // =========================

    addBloom<<<impl->gridSize, impl->blockSize>>>(
        impl->d_normalizedBuffer,
        impl->d_bloomBuffer,
        impl->d_finalHDRBuffer,
        impl->width,
        impl->height,
        impl->bloomStrength
    );
}

int Renderer::getFrameNumber(){
    return impl->sampleCount;
}
void Renderer::cleanup()
{
    cudaFree(impl->d_accumBuffer);

    cudaFree(impl->d_normalizedBuffer);

    cudaFree(impl->d_brightBuffer);

    cudaFree(impl->d_bloomBuffer);

    cudaFree(impl->d_tempBuffer);

    cudaFree(impl->d_finalHDRBuffer);

    cudaFree(impl->d_finalBuffer);

    cudaFree(impl->d_lvl1);

    cudaFree(impl->d_lvl2);

}

void Renderer::renderFrame()
{
    // =========================
    // Accumulate one sample
    // =========================

    renderKernel<<<impl->gridSize, impl->blockSize>>>(
        impl->gpuScene,
        impl->d_accumBuffer,
        impl->width,
        impl->height,
        impl->sampleCount
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "renderKernel error: " << cudaGetErrorString(err) << std::endl;
    }

    impl->sampleCount++;

    // =========================
    // Normalize accumulation
    // =========================

    normalizeKernel<<<impl->gridSize, impl->blockSize>>>(
        impl->d_accumBuffer,
        impl->d_normalizedBuffer,
        impl->sampleCount,
        impl->width,
        impl->height
    );

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "normalizeKernel error: " << cudaGetErrorString(err) << std::endl;
    }

    // =========================
    // Bloom
    // =========================

    applyBloom();

    // =========================
    // MAP OPENGL TEXTURE
    // =========================

    cudaGraphicsMapResources(
        1,
        &impl->cudaTextureResource
    );

    cudaArray_t textureArray;

    cudaGraphicsSubResourceGetMappedArray(
        &textureArray,
        impl->cudaTextureResource,
        0,
        0
    );

    // =========================
    // CREATE CUDA SURFACE
    // =========================

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = textureArray;

    cudaSurfaceObject_t surface = 0;

    cudaCreateSurfaceObject(
        &surface,
        &desc
    );

    // =========================
    // Tonemap + RGB8 conversion
    // =========================

    finalizeImage<<<impl->gridSize, impl->blockSize>>>(
        impl->d_finalHDRBuffer,
        surface,
        impl->width,
        impl->height,
        impl->exposure
    );
    
    // =========================
    // CLEANUP
    // =========================

    cudaDestroySurfaceObject(surface);

    cudaGraphicsUnmapResources(
        1,
        &impl->cudaTextureResource
    );
    
}