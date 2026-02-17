#ifndef __RT_ISICG_TRIANGLE_MESH__
#define __RT_ISICG_TRIANGLE_MESH__

#include "aabb.hpp"
#include "base_object.hpp"
#include "bvh.hpp"
#include "geometry/triangle_mesh_geometry.hpp"
#include <vector>

namespace RT
{
	class MeshTriangle : public BaseObject
	{
		friend class TriangleMeshGeometry;

	  public:
		MeshTriangle() = delete;
		MeshTriangle( const std::string & p_name ) : BaseObject( p_name ) {}
		virtual ~MeshTriangle() = default;

		const size_t getNbTriangles() const { return _triangles.size(); }
		const size_t getNbVertices() const { return _vertices.size(); }

		inline void addTriangle( const unsigned int p_v0, const unsigned int p_v1, const unsigned int p_v2 )
		{
			_triangles.emplace_back( TriangleMeshGeometry( p_v0, p_v1, p_v2, this ) );
		}
		inline void addVertex( const float p_x, const float p_y, const float p_z )
		{
			_vertices.emplace_back( p_x, p_y, p_z );
			_aabb.extend( Vec3f( p_x, p_y, p_z ) );
		}
		inline void addNormal( const float p_x, const float p_y, const float p_z )
		{
			_normals.emplace_back( p_x, p_y, p_z );
		}
		inline void addUV( const float p_u, const float p_v ) { _uvs.emplace_back( p_u, p_v ); }

		inline const AABB & getAABB() const { return _aabb; }

		inline const std::vector<TriangleMeshGeometry> getTriangles() const { return _triangles; }

		inline void buildBVH() { _bvh.build( &_triangles ); }

		inline BVH getBVH() const{return _bvh;}
		inline std::vector<Vec3f>	 getVertices() const{return _vertices;}
		inline std::vector<Vec3f> getNormals() const{return _normals;}
		inline std::vector<Vec2f>	 getUVS() const{return _uvs;}
		
		// Check for nearest intersection between p_tMin and p_tMax : if found fill p_hitRecord.
		bool intersect( const Ray & p_ray,
						const float p_tMin,
						const float p_tMax,
						HitRecord & p_hitRecord ) const override;

		// Check for any intersection between p_tMin and p_tMax.
		bool intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const override;
		ObjectType getType() const override { return ObjectType::TriangleMesh;}
	  private:
		std::vector<Vec3f>				  _vertices;
		std::vector<Vec3f>				  _normals;
		std::vector<Vec2f>				  _uvs;
		std::vector<TriangleMeshGeometry> _triangles;
		AABB							  _aabb;
		BVH								  _bvh;
	};
} // namespace RT

#endif // __RT_ISICG_TRIANGLE_MESH__
