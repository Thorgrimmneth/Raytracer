#include "fill_buffers.cuh"

GLOBAL
void initFloat3Buffer(float3* buffer, int count, float3 value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}

GLOBAL
void initFloat4Buffer(float4* buffer, int count, float4 value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}

GLOBAL
void initFloatBuffer(float* buffer, int count, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}

GLOBAL
void initBoolBuffer(bool* buffer, int count, bool value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}

GLOBAL
void initIntBuffer(int* buffer, int count, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}

GLOBAL
void initMaterialBuffer(MaterialType* buffer, int count, MaterialType value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    buffer[idx] = value;
}