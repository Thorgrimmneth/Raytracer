#include "post_treatment.cuh"

#include <cstdio>
#include <fstream>

GLOBAL
void extractBright(float3 *bright, float3 *normalize, float3 *out, int width, int height, float threshold, float invSampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;


    float3 c = bright[idx];
    c = c * invSampleCount;
    normalize[idx] = c;
    float maxChannel = fmaxf(c.x, fmaxf(c.y, c.z));
    out[idx] = (maxChannel > threshold) ? c : make_float3(0.f);
}

GLOBAL
void downsample(float3 *input, float3 *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int newWidth = width / 2;
    int newHeight = height / 2;

    if (x >= newWidth || y >= newHeight)
        return;

    int baseX = x * 2;
    int baseY = y * 2;

    int idx00 = baseY * width + baseX;
    int idx10 = baseY * width + baseX + 1;
    int idx01 = (baseY + 1) * width + baseX;
    int idx11 = (baseY + 1) * width + baseX + 1;

    output[y * newWidth + x] = (input[idx00] + input[idx10] + input[idx01] + input[idx11]) * 0.25f;
}

template <int RADIUS>
GLOBAL void blurHorizontal(const float3 *__restrict__ input, float3 *__restrict__ output, int width, int height)
{
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    __shared__ float3 tile[BLOCK_Y][BLOCK_X + 2 * RADIUS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Chargement complet du tile + halo
    for (int sx = tx; sx < BLOCK_X + 2 * RADIUS; sx += BLOCK_X)
    {
        int gx = blockIdx.x * BLOCK_X + sx - RADIUS;

        gx = max(0, min(gx, width - 1));

        if (y < height)
            tile[ty][sx] = input[y * width + gx];
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    constexpr float weights[] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    float3 result = tile[ty][tx + RADIUS] * weights[0];

#pragma unroll
    for (int i = 1; i <= RADIUS; ++i)
    {
        result += tile[ty][tx + RADIUS - i] * weights[i];
        result += tile[ty][tx + RADIUS + i] * weights[i];
    }

    output[y * width + x] = result;
}

template <int RADIUS>
GLOBAL void blurVertical(const float3 *__restrict__ input, float3 *__restrict__ output, int width, int height)
{
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    __shared__ float3 tile[BLOCK_Y + 2 * RADIUS][BLOCK_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;

    // Chargement complet du tile + halo
    for (int sy = ty; sy < BLOCK_Y + 2 * RADIUS; sy += BLOCK_Y)
    {
        int gy = blockIdx.y * BLOCK_Y + sy - RADIUS;

        gy = max(0, min(gy, height - 1));

        if (x < width)
            tile[sy][tx] = input[gy * width + x];
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    constexpr float weights[] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};

    float3 result = tile[ty + RADIUS][tx] * weights[0];

#pragma unroll
    for (int i = 1; i <= RADIUS; ++i)
    {
        result += tile[ty + RADIUS - i][tx] * weights[i];
        result += tile[ty + RADIUS + i][tx] * weights[i];
    }

    output[y * width + x] = result;
}

GLOBAL
void upsampleAdd(float3 *__restrict__ lowRes, float3 *highRes, int lowWidth, int lowHeight, int highWidth,
                 float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= highWidth || y >= lowHeight * 2)
        return;

    float gx = 0.5f * x - 0.25f;
    float gy = 0.5f * y - 0.5f;

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

    float3 a = c00 + tx * (c10 - c00);
    float3 b = c01 + tx * (c11 - c01);
    float3 c = a + ty * (b - a);

    highRes[y * highWidth + x] += c * strength;
}

void applyMultiScaleBloom(float3 *d_bright, float3 *d_temp, float3 *d_lvl1, float3 *d_lvl2, int w1, int h1, int w2,
                          int h2, int width, int height)
{
    dim3 block(16, 16);

    dim3 grid1((w1 + 15) / 16, (h1 + 15) / 16);

    downsample<<<grid1, block>>>(d_bright, d_lvl1, width, height);

    blurHorizontal<4><<<grid1, block>>>(d_lvl1, d_temp, w1, h1);
    blurVertical<4><<<grid1, block>>>(d_temp, d_lvl1, w1, h1);

    dim3 grid2((w2 + 15) / 16, (h2 + 15) / 16);

    downsample<<<grid2, block>>>(d_lvl1, d_lvl2, w1, h1);

    blurHorizontal<2><<<grid2, block>>>(d_lvl2, d_temp, w2, h2);
    blurVertical<2><<<grid2, block>>>(d_temp, d_lvl2, w2, h2);

    upsampleAdd<<<grid1, block>>>(d_lvl2, d_lvl1, w2, h2, w1, 1.0f);

    upsampleAdd<<<dim3((width + 15) / 16, (height + 15) / 16), block>>>(d_lvl1, d_bright, w1, h1, width, 1.0f);
}

GLOBAL
void addBloom(float3 *hdr, float3 *bloom, float3 *out, int width, int height, float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    out[idx] = hdr[idx] + bloom[idx] * strength;
}

GLOBAL
void normalizeKernel(float3 *accum, float3 *normalized, int sampleCount, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    normalized[idx] = accum[idx] / (float)sampleCount;
}

GLOBAL
void finalizeImage(float3 *hdr, cudaSurfaceObject_t surface, int width, int height, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    float3 &c = hdr[idx];

    // Reinhard tonemap
    c = (c * exposure) / (make_float3(1.f) + c * exposure);

    // Gamma correction
    c = make_float3(sqrtf(fmaxf(c.x, 0.f)), sqrtf(fmaxf(c.y, 0.f)), sqrtf(fmaxf(c.z, 0.f)));

    uchar4 pixel = make_uchar4((unsigned char)(255.f * fminf(c.x, 1.f)), (unsigned char)(255.f * fminf(c.y, 1.f)),
                               (unsigned char)(255.f * fminf(c.z, 1.f)), 255);

    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

GLOBAL
void finalizeImageV2(float3 *hdr, float3 *bloom, float3 *outCompare, cudaSurfaceObject_t surface, int width, int height, float exposure, float bloomStrength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    int idx = y * width + x;

    float3 hdrBloom = bloom[idx] * bloomStrength + hdr[idx];
    // apply bloom
    outCompare[idx] = hdrBloom;

    // finalize image
    float3 &c = hdr[idx];
    c = hdrBloom;
    // Reinhard tonemap
    c = (c * exposure) / (make_float3(1.f) + c * exposure);

    // Gamma correction
    c = make_float3(sqrtf(fmaxf(c.x, 0.f)), sqrtf(fmaxf(c.y, 0.f)), sqrtf(fmaxf(c.z, 0.f)));

    uchar4 pixel = make_uchar4((unsigned char)(255.f * fminf(c.x, 1.f)), (unsigned char)(255.f * fminf(c.y, 1.f)),
                               (unsigned char)(255.f * fminf(c.z, 1.f)), 255);

    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

GLOBAL
void finalizeImageV2NoRender(float3 *hdr, float3 *bloom, float3 *outCompare, int width, int height, float exposure, float bloomStrength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    int idx = y * width + x;

    float3 hdrBloom = bloom[idx] * bloomStrength + hdr[idx];
    // apply bloom
    outCompare[idx] = hdrBloom;

    // finalize image
    float3 &c = hdr[idx];
    c = hdrBloom;
    // Reinhard tonemap
    c = (c * exposure) / (make_float3(1.f) + c * exposure);

    // Gamma correction
    c = make_float3(sqrtf(fmaxf(c.x, 0.f)), sqrtf(fmaxf(c.y, 0.f)), sqrtf(fmaxf(c.z, 0.f)));

    uchar4 pixel = make_uchar4((unsigned char)(255.f * fminf(c.x, 1.f)), (unsigned char)(255.f * fminf(c.y, 1.f)),
                               (unsigned char)(255.f * fminf(c.z, 1.f)), 255);
}