#pragma once

#include "../objectsUtils/aabb.cuh"

class TriangleMesh;

struct TriangleMeshGeometry{
    float4 m0;
    float4 m1;
    float4 m2;

    int i0;
    int i1;
    int i2;

    

    TriangleMeshGeometry() = default;
    TriangleMeshGeometry(int index0, int index1, int index2, const float3* vertices){
            i0 = index0;
            i1 = index1;
            i2 = index2;
            const float3 v0 = vertices[index0];
            const float3 v1 = vertices[index1];
            const float3 v2 = vertices[index2];

            const float3 e0 = v0 - v2;
            const float3 e1 = v1 - v2;
            const float3 n  = cross(e0, e1);

            const float n2 = dot(n, n);

            if (n2 <= 1e-20f)
            {
                m0 = make_float4(0.f, 0.f, 0.f, 0.f);
                m1 = make_float4(0.f, 0.f, 0.f, 0.f);
                m2 = make_float4(0.f, 0.f, 0.f, 0.f);
                return;
            }

            const float invN2 = 1.0f / n2;

            const float3 rowU = cross(e1, n) * invN2;
            const float3 rowV = cross(n, e0) * invN2;
            const float3 rowZ = n * invN2;

            const float tu = -dot(rowU, v2);
            const float tv = -dot(rowV, v2);
            const float tz = -dot(rowZ, v2);

            // Si tu utilises l'intersection:
            // Oz = dot(m0.xyz, O) + m0.w
            // Ou = dot(m1.xyz, O) + m1.w
            // Ov = dot(m2.xyz, O) + m2.w
            // t  = -Oz / Dz
            // u  = Ou + t * Du
            // v  = Ov + t * Dv

            m0 = make_float4(rowZ.x, rowZ.y, rowZ.z, tz);
            m1 = make_float4(rowU.x, rowU.y, rowU.z, tu);
            m2 = make_float4(rowV.x, rowV.y, rowV.z, tv);
        };

    __device__ __forceinline__
    bool intersect(
            const Ray& ray,
            float tMin,
            float tMax,
            float& t,
            float2& uv) const
    {   
        
        const float Oz = dot3f4(m0, ray.origin) + m0.w;
        const float Ou = dot3f4(m1, ray.origin) + m1.w;
        const float Ov = dot3f4(m2, ray.origin) + m2.w;

        const float Dz = dot3f4(m0, ray.direction);
        const float Du = dot3f4(m1, ray.direction);
        const float Dv = dot3f4(m2, ray.direction);

        if (fabsf(Dz) < 1e-8f)
            return false;

        const float tTemp = -Oz / Dz;

        if (tTemp <= tMin || tTemp >= tMax)
            return false;

        const float u = Ou + tTemp * Du;
        const float v = Ov + tTemp * Dv;

        if (u < 0.0f || v < 0.0f || u + v > 1.0f)
            return false;

        t = tTemp;
        uv.x = u;
        uv.y = v;

        return true;
    }

    __device__
    const float3 computeSmoothNormal( const float2 & p_uv, const float3* normals ) const;
};