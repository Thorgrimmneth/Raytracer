#pragma once

#include <curand_kernel.h>

#include "../utils/macro.cuh"
#include "../utils/op.cuh"


GLOBAL
void extractBright(float3 *hdr, float3 *bright, int width, int height, float threshold);

GLOBAL
void downsample(float3 *input, float3 *output, int width, int height);

GLOBAL
void upsampleAdd(float3 *lowRes, float3 *highRes, int lowWidth, int lowHeight, int highWidth, float strength);

void applyMultiScaleBloom(float3 *d_bright, float3 *d_temp, float3 *d_lvl1, float3 *d_lvl2, int w1, int h1, int w2,
                          int h2, int width, int height);

GLOBAL
void addBloom(float3 *hdr, float3 *bloom, float3 *out, int width, int height, float strength);

GLOBAL
void normalizeKernel(float3 *accum, float3 *normalized, int sampleCount, int width, int height);

GLOBAL
void finalizeImage(float3 *hdr, cudaSurfaceObject_t surface, int width, int height, float exposure);