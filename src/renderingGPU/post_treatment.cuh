#pragma once

#include "utils/op.cuh"
#include <curand_kernel.h>

__global__
void extractBright(float3* hdr,
                   float3* bright,
                   int width,
                   int height,
                   float threshold);

__global__
void downsample(float3* input,
                float3* output,
                int width,
                int height);

__global__
void blurHorizontal(float3* input,
                    float3* output,
                    int width,
                    int height);

__global__
void blurVertical(float3* input,
                  float3* output,
                  int width,
                  int height);

__global__
void upsampleAdd(float3* lowRes,
                 float3* highRes,
                 int lowWidth,
                 int lowHeight,
                 int highWidth,
                 float strength);

void applyMultiScaleBloom(float3* d_bright,
                          float3* d_temp,
                          float3* d_lvl1, float3* d_lvl2,
                          int w1, int h1, int w2, int h2,
                          int width,
                          int height);

__global__
void addBloom(float3* hdr,
              float3* bloom,
              float3* out,
              int width,
              int height,
              float strength);

__global__
void normalizeKernel(
    float3* accum,
    float3* normalized,
    int sampleCount,
    int width,
    int height);
       
__global__
void finalizeImage(
    float3* hdr,
    cudaSurfaceObject_t surface,
    int width,
    int height,
    float exposure);