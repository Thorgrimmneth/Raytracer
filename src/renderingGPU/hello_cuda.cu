#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include "cuda_scene.cuh"
#include "../scene.hpp"
#include "renderingGPU/cuda_whitted_integrator.cuh"
#include "renderingGPU/cuda_direct_lighting_integrator.cuh"
#include <curand_kernel.h>

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
    unsigned char* d_framebuffer,
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

    float3 finalColor = make_float3(0.f, 0.f, 0.f);

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
    WhittedIntegrator integrator;
    for (int s = 0; s < nbSample; s++)
    {
        float sx = (x + curand_uniform(&localState)) / (float)(width  - 1);
        float sy = (y + curand_uniform(&localState)) / (float)(height - 1);

        float3 rayTarget = topLeft + sx * viewportU - sy * viewportV;
        float3 direction = normalize(rayTarget - camPos);

        float time = curand_uniform(&localState);

        Ray ray(camPos, direction, time);
        HitRecord hit;
        float3 color = integrator.Li(gpuScene, ray, 0, 1e20f, &localState);

        finalColor += color;
    }

    finalColor /= (float)nbSample;

    // ===== Gamma correction =====
    finalColor = make_float3(
        sqrtf(finalColor.x),
        sqrtf(finalColor.y),
        sqrtf(finalColor.z)
    );

    int fbIndex = pixelIndex * 3;

    d_framebuffer[fbIndex + 0] = (unsigned char)(255.f * clamp(finalColor.x,0.f,1.f));
    d_framebuffer[fbIndex + 1] = (unsigned char)(255.f * clamp(finalColor.y,0.f,1.f));
    d_framebuffer[fbIndex + 2] = (unsigned char)(255.f * clamp(finalColor.z,0.f,1.f));

    rngStates[pixelIndex] = localState;
}


unsigned char* launchHelloCUDA(const RT::Scene& scene,
                               const int nbSample,
                               const int width,
                               const int height)
{
    // ===== Upload scene =====
    CudaScene gpuScene = uploadSceneToGPU(scene);

    size_t bufferSize = width * height * 3 * sizeof(unsigned char);

    unsigned char* d_framebuffer;
    cudaMalloc(&d_framebuffer, bufferSize);

    // ===== Launch config =====
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    curandState* d_rngStates;
    cudaMalloc(&d_rngStates, width * height * sizeof(curandState));

    initRNG<<<gridSize, blockSize>>>(d_rngStates, width, height);
    cudaDeviceSynchronize(); // important avant benchmark

    // ===== Benchmark setup =====
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ===== Render =====
    renderKernel<<<gridSize, blockSize>>>(
        gpuScene,
        d_framebuffer,
        d_rngStates,
        nbSample,
        width,
        height
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Render time: %.3f ms\n", milliseconds/1000.f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ===== Error check =====
    cudaError_t errSync = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();

    if (errSync != cudaSuccess)
        printf("Sync error: %s\n", cudaGetErrorString(errSync));

    if (errAsync != cudaSuccess)
        printf("Async error: %s\n", cudaGetErrorString(errAsync));

    // ===== Copy back =====
    unsigned char* h_framebuffer = new unsigned char[width * height * 3];

    cudaMemcpy(h_framebuffer,
               d_framebuffer,
               bufferSize,
               cudaMemcpyDeviceToHost);

    cudaFree(d_framebuffer);
    cudaFree(d_rngStates);

    return h_framebuffer;
}
