#ifndef __RT_ISICG_TRIANGLE_GEOMETRY__
#define __RT_ISICG_TRIANGLE_GEOMETRY__

#include "aabb.hpp"
#include "base_geometry.hpp"

namespace RT
{
	class MeshTriangle;

	class TriangleMeshGeometry : public BaseGeometry
	{
	  public:
		TriangleMeshGeometry()			= delete;
		virtual ~TriangleMeshGeometry() = default;

		TriangleMeshGeometry( const unsigned int p_v0,
							  const unsigned int p_v1,
							  const unsigned int p_v2,
							  MeshTriangle *	 p_refMesh );

		bool intersect( const Ray & p_ray, float & p_t, Vec2f & p_uv ) const;

		inline const Vec3f & getFaceNormal() const { return _faceNormal; }

		inline const AABB & getAABB() const { return _aabb; }

		inline const MeshTriangle * getRefMesh() const { return _refMesh; }

		const Vec3f computeSmoothNormal( const Vec2f & p_uv ) const;

		inline float getArea() const { return _area; };

		const Vec3f samplePoint() const;
		inline unsigned int getV0() const { return _v0; }
		inline unsigned int getV1() const { return _v1; }
		inline unsigned int getV2() const { return _v2; }

	  private:
		MeshTriangle * _refMesh;
		union
		{
			struct
			{
				unsigned int _v0, _v1, _v2;
			};
			unsigned int _v[ 3 ] = { 0, 0, 0 };
		};

		Vec3f _faceNormal;
		AABB  _aabb;
		float _area = 0.f;
	};
} // namespace RT

#endif // __RT_ISICG_TRIANGLE_GEOMETRY__
