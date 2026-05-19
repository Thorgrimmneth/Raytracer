#include "triangle_mesh_geometry.cuh"
#include "triangle_mesh.cuh"

__device__
bool TriangleMeshGeometry::intersect(
    const Ray& ray,
    float tMin,
    float tMax,
    float& t,
    float2& uv,
    const float3* __restrict__ vertices) const
{
    const float3 o = ray.origin;
    const float3 d = ray.direction;

    const float3 v0 = vertices[i0];
    const float3 v1 = vertices[i1];
    const float3 v2 = vertices[i2];

    const float3 edge1 = v1 - v0;
    const float3 edge2 = v2 - v0;

    const float3 pvec = cross(d, edge2);
    const float det = dot(edge1, pvec);

    constexpr float epsilon = 1.e-8f;

    if (fabsf(det) < epsilon)
        return false;

    const float invDet = 1.0f / det;

    const float3 tvec = o - v0;
    const float u = dot(tvec, pvec) * invDet;

    if (u < 0.0f || u > 1.0f)
        return false;

    const float3 qvec = cross(tvec, edge1);
    const float v = dot(d, qvec) * invDet;

    if (v < 0.0f || u + v > 1.0f)
        return false;

    const float candidateT = dot(edge2, qvec) * invDet;

    if (candidateT <= tMin || candidateT >= tMax)
        return false;

    t = candidateT;
    uv = make_float2(u, v);

    return true;
}

__device__
const float3 TriangleMeshGeometry::computeSmoothNormal( const float2 & p_uv, const float3* normals ) const
{
	const float3 & n0 = normals[ i0 ];
	const float3 & n1 = normals[ i1 ];
	const float3 & n2 = normals[ i2 ];
	float		  u	 = p_uv.x;
	float		  v	 = p_uv.y;
	return ( 1 - u - v ) * n0 + u * n1 + v * n2;
}