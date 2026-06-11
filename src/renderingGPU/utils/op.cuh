#pragma once

// ============================================================
// CUDA float3 / float4 helpers
// Version conservative : garde le comportement du fichier original
// ============================================================

#include "macro.cuh"
#include <cuda_runtime.h>
#include <math.h>
// ============================================================
// Constructors / conversions
// ============================================================

HD_INLINE float3 make_float3(const float a)
{
    return make_float3(a, a, a);
}

HD_INLINE float3 make_float3(const float4 a)
{
    return make_float3(a.x, a.y, a.z);
}

HD_INLINE float4 float4f(const float a)
{
    return make_float4(a, a, a, 0.f);
}

HD_INLINE float3 toFloat3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}

HD_INLINE float4 toFloat4(const float3& a)
{
    return make_float4(a.x, a.y, a.z, 0.f);
}

HD_INLINE float4 make_float4(const float3& a, const float& b)
{
    return make_float4(a.x, a.y, a.z, b);
}

// ============================================================
// float3 operators
// ============================================================

HD_FORCEINLINE float3 operator+(const float3& a, const float3& b)
{
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

HD_FORCEINLINE float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

HD_FORCEINLINE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

HD_FORCEINLINE float3 operator-(const float4& a, const float3& b)
{
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

HD_FORCEINLINE float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

HD_FORCEINLINE float4 operator-(const float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, a.w);
}

// Garde l'overload original non-const pour ne pas changer la résolution d'overload.
HD_FORCEINLINE float3 operator*(const float3& a, float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

HD_FORCEINLINE float3& operator*=(float3& a, const float3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

HD_FORCEINLINE float3 operator*(const float3& a, const float b)
{
    return make_float3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

HD_FORCEINLINE float3 operator*(const float3& a, const float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

HD_FORCEINLINE float3 operator*(const float a, const float3& b)
{
    return make_float3(
        a * b.x,
        a * b.y,
        a * b.z
    );
}

HD_FORCEINLINE float3 operator/(const float3& a, const float b)
{
    const float inv = 1.0f / b;

    return make_float3(
        a.x * inv,
        a.y * inv,
        a.z * inv
    );
}

HD_FORCEINLINE float3 operator/(const float3& a, const float3& b)
{
    return make_float3(
        a.x * (1.0f / b.x),
        a.y * (1.0f / b.y),
        a.z * (1.0f / b.z)
    );
}

HD_FORCEINLINE float3 operator/=(float3& a, const float b)
{
    const float inv = 1.0f / b;
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    return a;
}

HD_INLINE bool operator==(const float3& a, const float3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// ============================================================
// float4 operators
// w reste forcé à 0.f comme dans ton fichier original
// ============================================================

HD_INLINE float4 operator+(const float4& a, const float4& b)
{
    return make_float4(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        0.f
    );
}

HD_INLINE float4 operator-(const float4& a, const float4& b)
{
    return make_float4(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
        0.f
    );
}

HD_INLINE float4 operator*(const float4& a, const float b)
{
    return make_float4(
        a.x * b,
        a.y * b,
        a.z * b,
        0.f
    );
}

HD_INLINE float4 operator*(const float a, const float4& b)
{
    return make_float4(
        a * b.x,
        a * b.y,
        a * b.z,
        0.f
    );
}

// ============================================================
// Vector math
// ============================================================

HD_FORCEINLINE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD_FORCEINLINE float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

HD_FORCEINLINE float3 normalize(const float3& a)
{
    const float len2 = dot(a, a);

    if (len2 > 0.0f)
    {
        return a * (1.0f / sqrtf(len2));
    }

    return make_float3(0.0f);
}

HD_FORCEINLINE float length2(const float3& a)
{
    return dot(a, a);
}

HD_FORCEINLINE float length(const float3& a)
{
    return sqrtf(length2(a));
}

HD_FORCEINLINE float length(const float4& a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

HD_FORCEINLINE float distance2(const float3& a, const float3& b)
{
    return length2(a - b);
}

HD_FORCEINLINE float distance(const float3& a, const float3& b)
{
    return length(a - b);
}

D_FORCEINLINE float3 abs(const float3& a)
{
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}
// ============================================================
// Scalar helpers
// ============================================================

HD_FORCEINLINE float clamp(const float x, const float lo, const float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

D_FORCEINLINE float3 clamp(const float3 x, const float3 lo, const float3 hi)
{
    return make_float3(
        clamp(x.x, lo.x, hi.x),
        clamp(x.y, lo.y, hi.y),
        clamp(x.z, lo.z, hi.z)
    );
}

HD_INLINE float3 lerp(const float3& a, const float3& b, const float c)
{
    return a * (1.f - c) + b * c;
}

HD_INLINE float lerp(const float a, const float b, const float c)
{
    return a * (1.f - c) + b * c;
}

HD_INLINE float getAxis(const float4& v, int axis)
{
    return axis == 0 ? v.x :
           axis == 1 ? v.y :
                       v.z;
}

// ============================================================
// Reflection / refraction
// ============================================================

HD_FORCEINLINE float3 reflect(const float3& a, const float3& b)
{
    return a - 2.0f * dot(a, b) * b;
}

HD_FORCEINLINE float3 refract(const float3& a, const float3& b, const float c)
{
    const float cosi = -dot(a, b);
    const float k = 1.0f - c * c * (1.0f - cosi * cosi);

    if (k < 0.0f)
    {
        return make_float3(0.0f);
    }

    return c * a + (c * cosi - sqrtf(k)) * b;
}

HD_FORCEINLINE bool refract(const float3& a, const float3& b, const float c, float3& out)
{
    const float cosi = -dot(a, b);
    const float k = 1.0f - c * c * (1.0f - cosi * cosi);

    if (k < 0.0f)
    {
        return false;
    }

    out = c * a + (c * cosi - sqrtf(k)) * b;
    return true;
}

// ============================================================
// Host-only min / max helpers
// ============================================================

H_INLINE float3 getMin(const float3& a, const float3& b)
{
    return make_float3(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z)
    );
}

H_INLINE float3 getMax(const float3& a, const float3& b)
{
    return make_float3(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z)
    );
}

H_INLINE float4 getMin(const float4& a, const float4& b)
{
    return make_float4(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z),
        0.f
    );
}

H_INLINE float4 getMax(const float4& a, const float4& b)
{
    return make_float4(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z),
        0.f
    );
}

// ============================================================
// Device-only helpers
// ============================================================

D_FORCEINLINE float smoothstep(float edge0, float edge1, float x)
{
    float t = (x - edge0) / (edge1 - edge0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

D_FORCEINLINE float saturate(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

D_FORCEINLINE float powerHeuristic(float pdfA, float pdfB)
{
    const float a = pdfA * pdfA;
    const float b = pdfB * pdfB;
    const float sum = a + b;

    return sum > 0.0f ? a / sum : 0.0f;
}

D_FORCEINLINE float dot3f4(const float4& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD inline float intBitsToFloat(int x)
{
#ifdef __CUDA_ARCH__
    return __int_as_float(x);
#else
    union {
        int i;
        float f;
    } u;

    u.i = x;
    return u.f;
#endif
}

HD inline int floatBitsToInt(float x)
{
#ifdef __CUDA_ARCH__
    return __float_as_int(x);
#else
    union {
        float f;
        int i;
    } u;

    u.f = x;
    return u.i;
#endif
}