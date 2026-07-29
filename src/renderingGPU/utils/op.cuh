#pragma once

// ============================================================
// CUDA float3 / float4 helpers
// Version conservative : garde le comportement du fichier original
// ============================================================

#include "simplified_def.cuh"
#include <cuda_runtime.h>
#include <math.h>
// ============================================================
// Constructors / conversions
// ============================================================

HD_INLINE float3 make_float3(const float a) { return make_float3(a, a, a); }

HD_INLINE float3 make_float3(const float4 a) { return make_float3(a.x, a.y, a.z); }

HD_INLINE float4 make_float4(const float3 &a, const float &b = 0.f) { return make_float4(a.x, a.y, a.z, b); }

HD_INLINE float4 make_float4(const float b) { return make_float4(b, b, b, b); }

HD_INLINE float4 make_float4(const float a, const float3 &b) { return make_float4(a, b.x, b.y, b.z); }

HD_INLINE float sign(float x)
{
    return (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
}

// ============================================================
// float3 operators
// ============================================================

HD_FORCEINLINE float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HD_FORCEINLINE float3 operator+(const float3 &a, const float4 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HD_FORCEINLINE float3 &operator+=(float3 &a, const float3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

HD_FORCEINLINE float4 &operator+=(float4 &a, const float4 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

HD_FORCEINLINE float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HD_FORCEINLINE float3 operator-(const float4 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HD_FORCEINLINE float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }

HD_FORCEINLINE float4 operator-(const float4 &a) { return make_float4(-a.x, -a.y, -a.z, a.w); }

// Garde l'overload original non-const pour ne pas changer la résolution d'overload.
HD_FORCEINLINE float3 operator*(const float3 &a, float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }

HD_FORCEINLINE float2 operator*(const float &a, const float2 &b) { return make_float2(a * b.x, a * b.y); }
HD_FORCEINLINE float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x - b.x, a.y - b.y); }
HD_FORCEINLINE float2 operator*(const float2 &a, const float &b) { return make_float2(a.x * b, a.y * b); }
HD_FORCEINLINE float2 operator*(const float2 &a, const float2 &b) { return make_float2(a.x * b.x, a.y * b.y); }


HD_FORCEINLINE float3 &operator*=(float3 &a, const float3 &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

HD_FORCEINLINE float4 &operator*=(float4 &a, const float3 &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

HD_FORCEINLINE float3 operator*(const float3 &a, const float b) { return make_float3(a.x * b, a.y * b, a.z * b); }

HD_FORCEINLINE float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

HD_FORCEINLINE float4 operator*(const float4&a, const float3 &b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, 0.f);
}

HD_FORCEINLINE float3 operator*(const float a, const float3 &b) { return make_float3(a * b.x, a * b.y, a * b.z); }

HD_FORCEINLINE float3 operator/(const float3 &a, const float b)
{
#ifdef __CUDA_ARCH__
    const float inv = __fdividef(1.0f, b);
#else
    const float inv = 1.0f / b;
#endif
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

HD_FORCEINLINE float4 operator/(const float4 &a, const float b)
{
#ifdef __CUDA_ARCH__
    const float inv = __fdividef(1.0f, b);
#else
    const float inv = 1.0f / b;
#endif
    return make_float4(a.x * inv, a.y * inv, a.z * inv, 0.f);
}

HD_FORCEINLINE float3 operator/(const float3 &a, const float3 &b)
{
#ifdef __CUDA_ARCH__
    return make_float3(a.x * __fdividef(1.0f, b.x), a.y * __fdividef(1.0f, b.y), a.z * __fdividef(1.0f, b.z));
#else
    return make_float3(a.x * (1.0f / b.x), a.y * (1.0f / b.y), a.z * (1.0f / b.z));
#endif
}

HD_FORCEINLINE float3 operator/=(float3 &a, const float b)
{
#ifdef __CUDA_ARCH__
    const float inv = __fdividef(1.0f, b);
#else
    const float inv = 1.0f / b;
#endif
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    return a;
}

HD_FORCEINLINE float4 operator/=(float4 &a, const float b)
{
#ifdef __CUDA_ARCH__
    const float inv = __fdividef(1.0f, b);
#else
    const float inv = 1.0f / b;
#endif
    a.x *= inv;
    a.y *= inv;
    a.z *= inv;
    return a;
}

HD_INLINE bool operator==(const float3 &a, const float3 &b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

// ============================================================
// float4 operators
// w reste forcé à 0.f comme dans ton fichier original
// ============================================================

HD_INLINE float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.f);
}

HD_INLINE float4 operator-(const float4 &a, const float4 &b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.f);
}

HD_INLINE float4 operator*(const float4 &a, const float b) { return make_float4(a.x * b, a.y * b, a.z * b, 0.f); }

HD_INLINE float4 operator*(const float a, const float4 &b) { return make_float4(a * b.x, a * b.y, a * b.z, 0.f); }

// ============================================================
// Vector math
// ============================================================

HD_FORCEINLINE float dot(const float3 &a, const float3 &b) { return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z)); }
HD_FORCEINLINE float dot(const float2 &a, const float2 &b) { return fmaf(a.x, b.x, a.y * b.y); }
HD_FORCEINLINE float dot4f3(const float4 &a, const float3 &b) { return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z)); }
HD_FORCEINLINE float dot(const float4 &a, const float4 &b) { return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z)); }

HD_FORCEINLINE float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(fmaf(a.y, b.z, -a.z * b.y), fmaf(a.z, b.x, -a.x * b.z), fmaf(a.x, b.y, -a.y * b.x));
}

HD_FORCEINLINE float3 normalize(const float3 &a)
{
    float len2 = dot(a, a);

    if (len2 > 0.0f)
    {
#ifdef __CUDA_ARCH__
        float invLen = rsqrtf(len2);
#else
        float invLen = 1.0f / sqrtf(len2);
#endif

        return a * invLen;
    }

    return make_float3(0.f);
}

HD_FORCEINLINE float length2(const float3 &a) { return dot(a, a); }

HD_FORCEINLINE float length(const float3 &a) { return sqrtf(length2(a)); }

HD_FORCEINLINE float length(const float2 &a) { return sqrtf(dot(a, a)); }

HD_FORCEINLINE float length(const float4 &a) { return sqrtf(fmaf(a.x, a.x, fmaf(a.y, a.y, a.z * a.z))); }

HD_FORCEINLINE float distance2(const float3 &a, const float3 &b) { return length2(a - b); }

HD_FORCEINLINE float distance(const float3 &a, const float3 &b) { return length(a - b); }

HD_FORCEINLINE float3 abs(const float3 &a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
// ============================================================
// Scalar helpers
// ============================================================

HD_FORCEINLINE float clamp(const float x, const float lo, const float hi) { return fminf(fmaxf(x, lo), hi); }

HD_FORCEINLINE float3 clamp(const float3 x, const float3 lo, const float3 hi)
{
    return make_float3(clamp(x.x, lo.x, hi.x), clamp(x.y, lo.y, hi.y), clamp(x.z, lo.z, hi.z));
}

HD_INLINE float3 lerp(const float3 &a, const float3 &b, float c)
{
    return make_float3(fmaf(c, b.x - a.x, a.x), fmaf(c, b.y - a.y, a.y), fmaf(c, b.z - a.z, a.z));
}

HD_INLINE float lerp(float a, float b, float c) { return fmaf(c, b - a, a); }

HD_INLINE float getAxis(const float3 &v, int axis) { return axis == 0 ? v.x : axis == 1 ? v.y : v.z; }

// ============================================================
// Reflection / refraction
// ============================================================

HD_FORCEINLINE float3 reflect(const float3 &a, const float3 &b)
{
    // Use FMA chain: k = -2 * dot(a, b)
    float k = -2.0f * dot(a, b);
    return make_float3(fmaf(k, b.x, a.x), fmaf(k, b.y, a.y), fmaf(k, b.z, a.z));
}

HD_FORCEINLINE float3 refract(const float3 &a, const float3 &b, const float c)
{
    const float cosi = -dot(a, b);
    // Optimize: 1.0 - c*c*(1 - cosi*cosi) = 1.0 - c*c + c*c*cosi*cosi
    const float k = fmaf(c * c, cosi * cosi, 1.0f - c * c);

    if (k < 0.0f)
    {
        return make_float3(0.0f);
    }

#ifdef __CUDA_ARCH__
    float t = fmaf(c, cosi, -sqrtf(k));
#else
    float t = c * cosi - sqrtf(k);
#endif
    return make_float3(fmaf(c, a.x, t * b.x), fmaf(c, a.y, t * b.y), fmaf(c, a.z, t * b.z));
}

HD_FORCEINLINE bool refract(const float3 &a, const float3 &b, const float c, float3 &out)
{
    const float cosi = -dot(a, b);
    // Optimize: 1.0 - c*c*(1 - cosi*cosi) = 1.0 - c*c + c*c*cosi*cosi
    const float k = fmaf(c * c, cosi * cosi, 1.0f - c * c);

    if (k < 0.0f)
    {
        return false;
    }

#ifdef __CUDA_ARCH__
    float t = fmaf(c, cosi, -sqrtf(k));
#else
    float t = c * cosi - sqrtf(k);
#endif
    out = c * a + t * b;
    return true;
}

// ============================================================
// Host-only min / max helpers
// ============================================================

H_INLINE float3 getMin(const float3 &a, const float3 &b)
{
    return make_float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

H_INLINE float3 getMax(const float3 &a, const float3 &b)
{
    return make_float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

H_INLINE float4 getMin(const float4 &a, const float4 &b)
{
    return make_float4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), 0.f);
}

H_INLINE float4 getMax(const float4 &a, const float4 &b)
{
    return make_float4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), 0.f);
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

D_FORCEINLINE float saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

D_FORCEINLINE float powerHeuristic(float pdfA, float pdfB)
{
    const float a = pdfA * pdfA;
    const float b = pdfB * pdfB;
    const float sum = a + b;

    return sum > 0.0f ? a / sum : 0.0f;
}

D_FORCEINLINE float dot3f4(const float4 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

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

struct Matrix3x3
{
    float3 row0;
    float3 row1;
    float3 row2;
    HD_INLINE static Matrix3x3 identity()
    {
        Matrix3x3 m;
        m.row0 = make_float3(1.f, 0.f, 0.f);
        m.row1 = make_float3(0.f, 1.f, 0.f);
        m.row2 = make_float3(0.f, 0.f, 1.f);
        return m;
    }

    HD_INLINE Matrix3x3 transpose() const
    {
        Matrix3x3 t;

        t.row0 = make_float3(row0.x, row1.x, row2.x);
        t.row1 = make_float3(row0.y, row1.y, row2.y);
        t.row2 = make_float3(row0.z, row1.z, row2.z);

        return t;
    }


};

HD_INLINE float3 transform(const Matrix3x3 &rotation, const float3 &v)
{
    return make_float3(dot(rotation.row0, v), dot(rotation.row1, v), dot(rotation.row2, v));
}

HD_INLINE float3 operator*(const Matrix3x3 &m, const float3 &v) { return transform(m, v); }

D_FORCEINLINE float3 transformPoint(const float transform[12], const float3& p)
{
    return make_float3(
        transform[0] * p.x + transform[1] * p.y + transform[2] * p.z + transform[3],
        transform[4] * p.x + transform[5] * p.y + transform[6] * p.z + transform[7],
        transform[8] * p.x + transform[9] * p.y + transform[10] * p.z + transform[11]
    );
}

D_FORCEINLINE float3 transformVector(const float transform[12], const float3& v)
{
    return make_float3(
        transform[0] * v.x + transform[1] * v.y + transform[2] * v.z,
        transform[4] * v.x + transform[5] * v.y + transform[6] * v.z,
        transform[8] * v.x + transform[9] * v.y + transform[10] * v.z
    );
}

HD_INLINE Matrix3x3 operator*(const Matrix3x3& a, const Matrix3x3& b)
{
    Matrix3x3 c;

    Matrix3x3 bt = b.transpose();

    c.row0 = make_float3(
        dot(a.row0, bt.row0),
        dot(a.row0, bt.row1),
        dot(a.row0, bt.row2));

    c.row1 = make_float3(
        dot(a.row1, bt.row0),
        dot(a.row1, bt.row1),
        dot(a.row1, bt.row2));

    c.row2 = make_float3(
        dot(a.row2, bt.row0),
        dot(a.row2, bt.row1),
        dot(a.row2, bt.row2));

    return c;
}