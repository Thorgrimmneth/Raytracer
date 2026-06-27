#pragma once

#include "op.cuh"
#include "defines.cuh"
#include "simplified_def.cuh"

struct Quaternion
{
    float w, x, y, z;
};

H_INLINE
Quaternion makeQuaternion(float w, float x, float y, float z)
{
    Quaternion q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

H_INLINE
Quaternion normalizeQuaternion(Quaternion q)
{
    float len = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    if (len == 0.f)
        return makeQuaternion(1.f, 0.f, 0.f, 0.f);

    float invLen = 1.f / len;

    return makeQuaternion(
        q.w * invLen,
        q.x * invLen,
        q.y * invLen,
        q.z * invLen
    );
}

H_INLINE
Quaternion conjugateQuaternion(Quaternion q)
{
    return makeQuaternion(q.w, -q.x, -q.y, -q.z);
}

H_INLINE
Quaternion multiplyQuaternion(Quaternion a, Quaternion b)
{
    return makeQuaternion(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.w + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    );
}

H_INLINE
Quaternion quaternionFromAxisAngle(float3 axis, float angle)
{
    float angleRad = angle * GPUPIf / 180.f;
    axis = normalize(axis);

    float halfAngle = 0.5f * angleRad;
    float s = sinf(halfAngle);

    return normalizeQuaternion(makeQuaternion(
        cosf(halfAngle),
        axis.x * s,
        axis.y * s,
        axis.z * s
    ));
}

H_INLINE
float3 rotateByQuaternionFast(float3 v, Quaternion q)
{
    q = normalizeQuaternion(q);

    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;

    return 2.f * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.f * s * cross(u, v);
}

H_INLINE
float3 transformPoint(
    float3 p,
    float3 scale,
    Quaternion rotation,
    float3 translation)
{
    p = make_float3(
        p.x * scale.x,
        p.y * scale.y,
        p.z * scale.z
    );

    p = rotateByQuaternionFast(p, rotation);

    p = p + translation;

    return p;
}

H_INLINE
float3 transformNormal(
    float3 n,
    Quaternion rotation)
{
    n = rotateByQuaternionFast(n, rotation);
    return normalize(n);
}