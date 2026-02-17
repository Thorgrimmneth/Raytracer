#include "triangle_mesh_geometry.hpp"
#include "objects/triangle_mesh.hpp"
#include "utils/random.hpp"

namespace RT
{
	TriangleMeshGeometry::TriangleMeshGeometry( const unsigned int p_v0,
												const unsigned int p_v1,
												const unsigned int p_v2,
												MeshTriangle *	   p_refMesh )
		: _v0( p_v0 ), _v1( p_v1 ), _v2( p_v2 ), _refMesh( p_refMesh )
	{
		_faceNormal = glm::normalize( glm::cross( _refMesh->_vertices[ p_v1 ] - _refMesh->_vertices[ p_v0 ],
												  _refMesh->_vertices[ p_v2 ] - _refMesh->_vertices[ p_v0 ] ) );
		_aabb.extend( _refMesh->_vertices[ p_v0 ] );
		_aabb.extend( _refMesh->_vertices[ p_v1 ] );
		_aabb.extend( _refMesh->_vertices[ p_v2 ] );
		_area = 0.5f
				* glm::length( glm::cross( _refMesh->_vertices[ _v1 ] - _refMesh->_vertices[ _v0 ],
										   _refMesh->_vertices[ _v2 ] - _refMesh->_vertices[ _v0 ] ) );
	}

	bool TriangleMeshGeometry::intersect( const Ray & p_ray, float & p_t, Vec2f & p_uv ) const
	{
		// [MT97] Tomas Moller and Ben Trumbore. Fast, minimum storage ray-triangle intersection. J. Graph. Tools, 2( 1
		// ) : 21 28, October 1997.
		const Vec3f & o	 = p_ray.getOrigin();
		const Vec3f & d	 = p_ray.getDirection();
		const Vec3f & v0 = _refMesh->_vertices[ _v0 ];
		const Vec3f & v1 = _refMesh->_vertices[ _v1 ];
		const Vec3f & v2 = _refMesh->_vertices[ _v2 ];

		float det;
		float inv_det;
		float epsilon = 1.e-6f;

		Vec3f edge1 = v1 - v0;
		Vec3f edge2 = v2 - v0;
		Vec3f pvec	= glm::cross( d, edge2 );
		det			= glm::dot( edge1, pvec );
		if ( det > -epsilon && det < epsilon ) return false;

		inv_det	   = 1.0 / det;
		Vec3f tvec = o - v0;
		float u	   = glm::dot( tvec, pvec ) * inv_det;
		if ( u < 0.f || u > 1.f ) return false;

		Vec3f qvec = glm::cross( tvec, edge1 );
		float v	   = glm::dot( d, qvec ) * inv_det;
		if ( v < 0.f || ( u + v ) > 1.f ) return false;

		p_t = glm::dot( edge2, qvec ) * inv_det;

		p_uv = Vec2f( u, v );
		return true;
	}

	// Compute a normal at a point using barycentric coordinates
	const Vec3f TriangleMeshGeometry::computeSmoothNormal( const Vec2f & p_uv ) const
	{
		const Vec3f & n0 = _refMesh->_normals[ _v0 ];
		const Vec3f & n1 = _refMesh->_normals[ _v1 ];
		const Vec3f & n2 = _refMesh->_normals[ _v2 ];
		float		  u	 = p_uv[ 0 ];
		float		  v	 = p_uv[ 1 ];
		return ( 1 - u - v ) * n0 + u * n1 + v * n2;
	}

	// Sample a point on the triangle using barycentric coordinates
	const Vec3f TriangleMeshGeometry::samplePoint() const
	{
		const Vec3f & v0 = _refMesh->_vertices[ _v0 ];
		const Vec3f & v1 = _refMesh->_vertices[ _v1 ];
		const Vec3f & v2 = _refMesh->_vertices[ _v2 ];
		float		  u	 = randomFloat();
		float		  v	 = randomFloat();
		if ( u + v > 1.f )
		{
			u = 1 - u;
			v = 1 - v;
		}
		return ( 1 - u - v ) * v0 + u * v1 + v * v2;
	}

} // namespace RT
