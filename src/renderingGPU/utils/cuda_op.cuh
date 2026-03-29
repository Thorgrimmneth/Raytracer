#pragma once

__host__ __device__
inline float3 make_float3(const float a)
{
    return make_float3(a, a, a);
}

__host__ __device__
inline float4 float4f(const float a)
{
    return make_float4(a, a, a,0.f);
}

__host__ __device__
inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__
inline float4 operator+(const float4& a, const float4& b)
{
    return make_float4(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        0.f
    );
}

__host__ __device__
inline float3 operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__
inline float4 operator-(const float4& a, const float4& b)
{
    return make_float4(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
        0.f
    );
}

__host__ __device__
inline float3 operator*(const float3& a, float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

__host__ __device__
inline float3 operator*=(const float3& a, const float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

__host__ __device__
inline float3 operator*(const float3& a, float b)
{
    return make_float3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}

__host__ __device__
inline float4 operator*(const float4& a, float b)
{
    return make_float4(
        a.x * b,
        a.y * b,
        a.z * b,
        0.f
    );
}

__host__ __device__
inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

__host__ __device__
inline float3 operator*(const float a, const float3& b)
{
    return make_float3(
        a * b.x,
        a * b.y,
        a * b.z
    );
}

__host__ __device__
inline float4 operator*(const float a, const float4& b)
{
    return make_float4(
        a * b.x,
        a * b.y,
        a * b.z,
        0.f
    );
}

__host__ __device__
inline float3 operator/(const float3& a, float b)
{
    return make_float3(
        a.x / b,
        a.y / b,
        a.z / b
    );
}

__host__ __device__
inline float3 operator/(const float3& a, const float3& b)
{
    return make_float3(
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    );
}

__host__ __device__
inline float3 operator/=(float3& a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}


__host__ __device__
inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__
inline float3 normalize(const float3& a)
{
    float len = sqrtf(dot(a,a));
    return (len > 0.f) ? a * (1.f / len) : make_float3(0.f);
}

__host__ __device__
inline float clamp(const float a, const float b, const float c)
{
    return min(max(a, b),c);
}

__host__ __device__
inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__
inline float3 lerp(const float3& a, const float3& b, const float c)
{
    return a * (1.f - c) + b * c;
}

__host__ __device__
inline float lerp(const float a, const float b, const float c)
{
    return a * (1.f - c) + b * c;
}

__host__ __device__
inline float length(const float3& a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline float distance(const float3& a, const float3& b)
{
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

__host__ __device__
inline float3 toFloat3(const float4& a)
{
    return make_float3(a.x, a.y, a.z);
}

__host__ __device__
inline float4 toFloat4(const float3& a)
{
    return make_float4(a.x, a.y, a.z, 0.f);
}

__host__ __device__
inline bool operator==(const float3& a, const float3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__
inline float3 reflect(const float3& a, const float3& b)
{
    return normalize(a - 2 * (dot(a, b)) * b);
}

__host__ __device__
inline float3 refract(const float3& a, const float3& b, const float c)
{
    float cosi = -dot(a, b);
    float k = 1.f - c * c * (1.f - cosi * cosi);

    if (k < 0.f)
        return make_float3(0.0f); // réflexion totale interne

    return c * a + (c * cosi - sqrtf(k)) * b;
}

__host__ __device__
inline bool refract(const float3& a, const float3& b, const float c, float3& out)
{
    float cosi = -dot(a, b);
    float k = 1.f - c * c * (1.f - cosi * cosi);

    if (k < 0.f)
        return false; // réflexion totale interne

    out = c * a + (c * cosi - sqrtf(k)) * b;
    return true;
}

__host__
inline float4 getMin(const float4& a, const float4& b){
    return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), 0.f);
}

__host__
inline float4 getMax(const float4& a, const float4& b){
    return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), 0.f);
}

__host__ __device__
inline float getAxis(const float4& v, int axis)
{
    return axis == 0 ? v.x :
           axis == 1 ? v.y :
                       v.z;
}

__host__ __device__
inline float4 make_float4(const float3& a, const float& b){
    return make_float4(a.x, a.y, a.z, b);
}

__device__ __forceinline__
float smoothstep(float edge0, float edge1, float x)
{
    float t = (x - edge0) / (edge1 - edge0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);   // clamp 0–1
    return t * t * (3.0f - 2.0f * t);
}

static __device__ float saturate(float x)
{
    return fminf(fmaxf(x, 0.f), 1.f);
}