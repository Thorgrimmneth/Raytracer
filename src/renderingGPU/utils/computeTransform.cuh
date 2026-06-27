#pragma once

#include "../objects/triangle_mesh.cuh"

inline void computeTransform(const MeshInstance& mesh, OptixInstance& instance)
{
    float4 q = mesh.rotation;
        float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
        float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
        float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
        float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;

        float r00 = 1.f - (yy + zz), r01 = xy - wz, r02 = xz + wy;
        float r10 = xy + wz, r11 = 1.f - (xx + zz), r12 = yz - wx;
        float r20 = xz - wy, r21 = yz + wx, r22 = 1.f - (xx + yy);

        float3 s = mesh.scale;
        float3 t = mesh.translation;

        instance.transform[0] = r00 * s.x;
        instance.transform[1] = r01 * s.y;
        instance.transform[2] = r02 * s.z;
        instance.transform[3] = t.x;
        instance.transform[4] = r10 * s.x;
        instance.transform[5] = r11 * s.y;
        instance.transform[6] = r12 * s.z;
        instance.transform[7] = t.y;
        instance.transform[8] = r20 * s.x;
        instance.transform[9] = r21 * s.y;
        instance.transform[10] = r22 * s.z;
        instance.transform[11] = t.z;
}