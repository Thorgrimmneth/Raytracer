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
float3 sunDirectionFromAngles(float elevation, float azimuth)
{
    elevation = elevation * GPUPIf / 180.f;
    azimuth = azimuth * GPUPIf / 180.f;
    float cosE = cos(elevation);

    return normalize(make_float3(-cosE * cos(azimuth), sin(elevation), -cosE * sin(azimuth)));
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
void initConstant(int width, int height){
    int c_nbBounces = 5;
    float c_earthRadius = 6360e3f;
    float3 c_sunDirection = sunDirectionFromAngles(50.f, 20.f);
    int c_skyColorSamples = 8;
    float c_hr = 7994.f;
    float c_hm = 1200.f;
    float3 c_betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    float3 c_betaM = float3f(21e-6f);
    float c_exposure = 20.f;
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

    for (int s = 0; s < nbSample; s++)
    {
        float sx = (x + curand_uniform(&localState)) / (float)(width  - 1);
        float sy = (y + curand_uniform(&localState)) / (float)(height - 1);

        float3 rayTarget = toFloat3(camera.topLeft + sx * camera.viewPortU - sy * camera.viewPortV);
        float3 direction = normalize(rayTarget - toFloat3(camera.cameraPos));

        float time = curand_uniform(&localState);

        Ray ray(toFloat3(camera.cameraPos), direction, time);
        float3 color = WhittedIntegrator::lighting(gpuScene, ray, 0, 1e20f, &localState);

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
                               const int height,
                                float elevation,
                                float azimuth)
{
    // ===== Upload scene =====
    printf("Start uploading the scene to the GPU\n");
    float3 sunDir = -sunDirectionFromAngles(elevation, azimuth);
    printf("%f, %f, %f\n", sunDir.x, sunDir.y, sunDir.z);
    CudaScene gpuScene = uploadSceneToGPU(scene, sunDir);
    printf("Done uploading the scene to the GPU\n");
    size_t bufferSize = width * height * 3 * sizeof(unsigned char);
    unsigned char* d_framebuffer;
    cudaMalloc(&d_framebuffer, bufferSize);
    printf("%i\n", gpuScene.bvhScene.nbNodes);
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
    initConstant(width, height);
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
    float seconds = milliseconds/1000.f;
    printf("Render time: %.3f ms\n", seconds);
    printf("Render time: %i minutes and %i s\n", (int)seconds/60, (int)seconds%60);

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

