#include "cuda_triangle_mesh_geometry.cuh"
#include "cuda_triangle_mesh.cuh"

__device__
bool TriangleMeshGeometry::intersect(const Ray &p_ray, float &p_t, float2 &p_uv, float3* vertices) const
{
    // [MT97] Tomas Moller and Ben Trumbore. Fast, minimum storage ray-triangle intersection. J. Graph. Tools, 2( 1
    // ) : 21 28, October 1997.
    const float3 &o = p_ray.origin;
    const float3 &d = p_ray.direction;
    const float3 &v0 = vertices[i0];
    const float3 &v1 = vertices[i1];
    const float3 &v2 = vertices[i2];

    float det;
    float inv_det;
    float epsilon = 1.e-6f;

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 pvec = cross(d, edge2);
    det = dot(edge1, pvec);
    if (det > -epsilon && det < epsilon)
        return false;

    inv_det = 1.0 / det;
    float3 tvec = o - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.f || u > 1.f)
        return false;

    float3 qvec = cross(tvec, edge1);
    float v = dot(d, qvec) * inv_det;
    if (v < 0.f || (u + v) > 1.f)
        return false;

    p_t = dot(edge2, qvec) * inv_det;

    p_uv = make_float2(u, v);
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