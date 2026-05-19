#include "post_treatment.cuh"

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

    normalized[idx] = accum[idx] / (float)sampleCount;
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