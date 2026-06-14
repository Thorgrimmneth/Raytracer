#pragma once

#include <curand_kernel.h>

#include "../utils/macro.cuh"
#include "../utils/op.cuh"


GLOBAL
void extractBright(float4 *hdr, float4 *bright, int width, int height, float threshold);

GLOBAL
void downsample(float4 *input, float4 *output, int width, int height);

GLOBAL
void upsampleAdd(float4 *lowRes, float4 *highRes, int lowWidth, int lowHeight, int highWidth, float strength);

void applyMultiScaleBloom(float4 *d_bright, float4 *d_temp, float4 *d_lvl1, float4 *d_lvl2, int w1, int h1, int w2,
                          int h2, int width, int height);

GLOBAL
void addBloom(float4 *hdr, float4 *bloom, float4 *out, int width, int height, float strength);

GLOBAL
void normalizeKernel(float4 *accum, float4 *normalized, int sampleCount, int width, int height);

GLOBAL
void finalizeImage(float4 *hdr, cudaSurfaceObject_t surface, int width, int height, float exposure);