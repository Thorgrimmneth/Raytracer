#pragma once

#include "defines.cuh"
#include "op.cuh"
#include "simplifiedDef.cuh"

struct Quaternion
{
    float w, x, y, z;
};

H_INLINE Quaternion makeQuaternion(float w, float x, float y, float z)
{
    Quaternion q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

H_INLINE Quaternion normalizeQuaternion(Quaternion q)
{
    float len = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    if (len == 0.f)
        return makeQuaternion(1.f, 0.f, 0.f, 0.f);

    float invLen = 1.f / len;

    return makeQuaternion(q.w * invLen, q.x * invLen, q.y * invLen, q.z * invLen);
}

H_INLINE Quaternion conjugateQuaternion(Quaternion q) { return makeQuaternion(q.w, -q.x, -q.y, -q.z); }

H_INLINE Quaternion multiplyQuaternion(Quaternion a, Quaternion b)
{
    return makeQuaternion(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

H_INLINE Quaternion quaternionFromAxisAngle(float3 axis, float angle)
{
    float angleRad = angle * GPUPIf / 180.f;
    axis = normalize(axis);

    float halfAngle = 0.5f * angleRad;
    float s = sinf(halfAngle);

    return normalizeQuaternion(makeQuaternion(cosf(halfAngle), axis.x * s, axis.y * s, axis.z * s));
}

H_INLINE float3 rotateByQuaternionFast(float3 v, Quaternion q)
{
    q = normalizeQuaternion(q);

    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;

    return 2.f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.f * s * cross(u, v);
}

H_INLINE float3 transformPoint(float3 p, float3 scale, Quaternion rotation, float3 translation)
{
    p = make_float3(p.x * scale.x, p.y * scale.y, p.z * scale.z);

    p = rotateByQuaternionFast(p, rotation);

    p = p + translation;

    return p;
}

H_INLINE float3 transformNormal(float3 n, Quaternion rotation)
{
    n = rotateByQuaternionFast(n, rotation);
    return normalize(n);
}

H_INLINE Matrix3x3 quaternionToMatrix(const Quaternion &q)
{
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;

    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;

    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    Matrix3x3 m;

    m.row0 = make_float3(1.f - 2.f * (yy + zz), 2.f * (xy - wz), 2.f * (xz + wy));

    m.row1 = make_float3(2.f * (xy + wz), 1.f - 2.f * (xx + zz), 2.f * (yz - wx));

    m.row2 = make_float3(2.f * (xz - wy), 2.f * (yz + wx), 1.f - 2.f * (xx + yy));

    return m;
}