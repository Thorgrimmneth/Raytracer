#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include "cuda_scene.cuh"
#include "../scene.hpp"
#include "integrators/cuda_direct_lighting_integrator.cuh"
#include "integrators/cuda_whitted_integrator.cuh"
#include <curand_kernel.h>
#include "camera/camera.cuh"

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
    float3* d_hdrBuffer,
    curandState* rngStates,
    int nbSample,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIndex = y * width + x;
    curandState localState = rngStates[pixelIndex];

    float3 finalColor = make_float3(0.f);

    for (int s = 0; s < nbSample; s++)
    {
        float sx = (x + curand_uniform(&localState)) / (float)(width  - 1);
        float sy = (y + curand_uniform(&localState)) / (float)(height - 1);

        float3 rayTarget = toFloat3(camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV);
        float3 direction = normalize(rayTarget - toFloat3(camera.cameraPos));

        Ray ray(toFloat3(camera.cameraPos), direction);
        finalColor += WhittedIntegrator::lighting(gpuScene, ray, 0, 1e20f, &localState);
    }

    finalColor /= (float)nbSample;

    d_hdrBuffer[pixelIndex] = finalColor;

    rngStates[pixelIndex] = localState;
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
                          int width,
                          int height)
{
    dim3 block(16,16);

    int w1 = width / 2;
    int h1 = height / 2;

    float3* d_lvl1;
    cudaMalloc(&d_lvl1, w1 * h1 * sizeof(float3));

    dim3 grid1((w1+15)/16,(h1+15)/16);

    downsample<<<grid1,block>>>(d_bright,d_lvl1,width,height);
    cudaDeviceSynchronize();

    blurHorizontal<<<grid1,block>>>(d_lvl1,d_temp,w1,h1);
    blurVertical<<<grid1,block>>>(d_temp,d_lvl1,w1,h1);
    cudaDeviceSynchronize();

    int w2 = w1 / 2;
    int h2 = h1 / 2;

    float3* d_lvl2;
    cudaMalloc(&d_lvl2, w2 * h2 * sizeof(float3));

    dim3 grid2((w2+15)/16,(h2+15)/16);

    downsample<<<grid2,block>>>(d_lvl1,d_lvl2,w1,h1);
    cudaDeviceSynchronize();

    blurHorizontal<<<grid2,block>>>(d_lvl2,d_temp,w2,h2);
    blurVertical<<<grid2,block>>>(d_temp,d_lvl2,w2,h2);
    cudaDeviceSynchronize();

    upsampleAdd<<<grid1,block>>>(d_lvl2,d_lvl1,w2,h2,w1,1.0f);
    cudaDeviceSynchronize();

    upsampleAdd<<<dim3((width+15)/16,(height+15)/16),block>>>(
        d_lvl1,d_bright,w1,h1,width,1.0f);
    cudaDeviceSynchronize();

    cudaFree(d_lvl1);
    cudaFree(d_lvl2);
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
void finalizeImage(float3* hdr,
                   unsigned char* out,
                   int width,
                   int height,
                   float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int fb  = idx * 3;

    float3 c = hdr[idx];

    c = (c * exposure) / (make_float3(1.f) + c * exposure);

    // gamma
    c = make_float3(
        sqrtf(fmaxf(c.x, 0.f)),
        sqrtf(fmaxf(c.y, 0.f)),
        sqrtf(fmaxf(c.z, 0.f))
    );

    out[fb + 0] = (unsigned char)(255.f * fminf(c.x, 1.f));
    out[fb + 1] = (unsigned char)(255.f * fminf(c.y, 1.f));
    out[fb + 2] = (unsigned char)(255.f * fminf(c.z, 1.f));
}

unsigned char* launchHelloCUDA(const RT::Scene& scene,
                               const int nbSample,
                               const int width,
                               const int height,
                               float sunDirx,
                                float sunDiry,
                            float sunDirz)
{
    float threshold = 10.0f;
    float bloomStrength = 0.25f;
    float exposure = 1.0f;
    float4 sunDir = make_float4(sunDirx, sunDiry, sunDirz, 0.f);
    CudaScene gpuScene = uploadSceneToGPU(scene, sunDir);

    size_t hdrBufferSize = width * height * sizeof(float3);
    size_t finalBufferSize = width * height * 3 * sizeof(unsigned char);

    float3* d_hdrBuffer;
    float3* d_brightBuffer;
    float3* d_outBuffer;
    unsigned char* d_finalBuffer;

    cudaMalloc(&d_hdrBuffer, hdrBufferSize);
    cudaMalloc(&d_brightBuffer, hdrBufferSize);
    cudaMalloc(&d_outBuffer, hdrBufferSize);
    cudaMalloc(&d_finalBuffer, finalBufferSize);

    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, width * height * sizeof(curandState));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    initRNG<<<gridSize, blockSize>>>(d_rngStates, width, height);
    cudaDeviceSynchronize();

    initConstant(width, height, sunDir);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    renderKernel<<<gridSize, blockSize>>>(
        gpuScene,
        d_hdrBuffer,
        d_rngStates,
        nbSample,
        width,
        height
    );
    cudaDeviceSynchronize();
    
    extractBright<<<gridSize, blockSize>>>(
        d_hdrBuffer,
        d_brightBuffer,
        width,
        height,
        threshold
    );
    cudaDeviceSynchronize();
    applyMultiScaleBloom(d_brightBuffer, d_outBuffer, width, height);
    addBloom<<<gridSize, blockSize>>>(
        d_hdrBuffer,
        d_brightBuffer,
        d_outBuffer,
        width,
        height,
        bloomStrength
    );
    cudaDeviceSynchronize();
    
    finalizeImage<<<gridSize, blockSize>>>(
        d_outBuffer,
        d_finalBuffer,
        width,
        height,
        exposure
    );
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Render time: %.3f s\n", milliseconds / 1000.f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t errSync = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();

    if (errSync != cudaSuccess)
        printf("Sync error: %s\n", cudaGetErrorString(errSync));

    if (errAsync != cudaSuccess)
        printf("Async error: %s\n", cudaGetErrorString(errAsync));

    unsigned char* h_framebuffer = new unsigned char[width * height * 3];

    cudaMemcpy(h_framebuffer,
               d_finalBuffer,
               finalBufferSize,
               cudaMemcpyDeviceToHost);

    cudaFree(d_hdrBuffer);
    cudaFree(d_brightBuffer);
    cudaFree(d_outBuffer);
    cudaFree(d_finalBuffer);
    cudaFree(d_rngStates);

    return h_framebuffer;
}

