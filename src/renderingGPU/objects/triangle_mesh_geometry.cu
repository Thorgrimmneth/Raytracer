#include "triangle_mesh_geometry.cuh"

__device__
const float3 TriangleMeshGeometry::computeSmoothNormal( const float2 & p_uv, const float3* normals ) const
{
	const float3 & n0 = normals[ i0];
	const float3 & n1 = normals[ i1 ];
	const float3 & n2 = normals[ i2 ];
	float		  u	 = p_uv.x;
	float		  v	 = p_uv.y;
	return ( 1 - u - v ) * n0 + u * n1 + v * n2;
}