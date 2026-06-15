#pragma once

#include "macro.cuh"

GLOBAL
void initFloat3Buffer(float3* buffer, int count, float3 value);

GLOBAL
void initFloat4Buffer(float4* buffer, int count, float4 value);

GLOBAL
void initFloatBuffer(float* buffer, int count, float value);

GLOBAL
void initBoolBuffer(bool* buffer, int count, bool value);

GLOBAL
void initIntBuffer(int* buffer, int count, int value);