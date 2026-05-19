#pragma once

// ============================================================
// CUDA float3 / float4 helpers
// Version conservative : garde le comportement du fichier original
// ============================================================

#ifndef HD_INLINE
#define HD_INLINE __host__ __device__ inline
#endif

#ifndef D_FORCEINLINE
#define D_FORCEINLINE __device__ __forceinline__
#endif

#ifndef H_INLINE
#define H_INLINE __host__ inline
#endif

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

HD_INLINE float3 operator+(const float3& a, const float3& b)
{
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

HD_INLINE float3 operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

HD_INLINE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

HD_INLINE float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// Garde l'overload original non-const pour ne pas changer la résolution d'overload.
HD_INLINE float3 operator*(const float3& a, float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

// Garde le comportement original : ce n'est pas un vrai *= in-place.
HD_INLINE float3 operator*=(const float3& a, const float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

HD_INLINE float3 operator*(const float3& a, const float b)
{
    return make_float3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

HD_INLINE float3 operator*(const float3& a, const float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

HD_INLINE float3 operator*(const float a, const float3& b)
{
    return make_float3(
        a * b.x,
        a * b.y,
        a * b.z
    );
}

HD_INLINE float3 operator/(const float3& a, const float b)
{
    return make_float3(
        a.x / b,
        a.y / b,
        a.z / b
    );
}

HD_INLINE float3 operator/(const float3& a, const float3& b)
{
    return make_float3(
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    );
}

HD_INLINE float3 operator/=(float3& a, const float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
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

HD_INLINE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD_INLINE float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

HD_INLINE float3 normalize(const float3& a)
{
    float len = sqrtf(dot(a, a));
    return (len > 0.f) ? a * (1.f / len) : make_float3(0.f);
}

HD_INLINE float length(const float3& a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

HD_INLINE float length2(const float3& a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

HD_INLINE float distance(const float3& a, const float3& b)
{
    return sqrtf(
        (a.x - b.x) * (a.x - b.x) +
        (a.y - b.y) * (a.y - b.y) +
        (a.z - b.z) * (a.z - b.z)
    );
}

// ============================================================
// Scalar helpers
// ============================================================

HD_INLINE float clamp(const float a, const float b, const float c)
{
    return min(max(a, b), c);
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
// reflect garde normalize(), indispensable avec ton code actuel
// ============================================================

HD_INLINE float3 reflect(const float3& a, const float3& b)
{
    return normalize(a - 2 * (dot(a, b)) * b);
}

HD_INLINE float3 refract(const float3& a, const float3& b, const float c)
{
    float cosi = -dot(a, b);
    float k = 1.f - c * c * (1.f - cosi * cosi);

    if (k < 0.f)
        return make_float3(0.0f);

    return c * a + (c * cosi - sqrtf(k)) * b;
}

HD_INLINE bool refract(const float3& a, const float3& b, const float c, float3& out)
{
    float cosi = -dot(a, b);
    float k = 1.f - c * c * (1.f - cosi * cosi);

    if (k < 0.f)
        return false;

    out = c * a + (c * cosi - sqrtf(k)) * b;
    return true;
}

// ============================================================
// Host-only min / max helpers
// ============================================================

H_INLINE float3 getMin(const float3& a, const float3& b)
{
    return make_float3(
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z)
    );
}

H_INLINE float3 getMax(const float3& a, const float3& b)
{
    return make_float3(
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z)
    );
}

H_INLINE float4 getMin(const float4& a, const float4& b)
{
    return make_float4(
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
        0.f
    );
}

H_INLINE float4 getMax(const float4& a, const float4& b)
{
    return make_float4(
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
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

static inline __device__ float saturate(float x)
{
    return fminf(fmaxf(x, 0.f), 1.f);
}

static inline __device__ float powerHeuristic(float pdfA, float pdfB)
{
    float a = pdfA * pdfA;
    float b = pdfB * pdfB;
    return a / (a + b);
}

D_FORCEINLINE float dot3f4(const float4& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}