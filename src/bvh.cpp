#include "bvh.hpp"
#include "geometry/triangle_mesh_geometry.hpp"
#include "utils/chrono.hpp"

#include <algorithm>

namespace RT
{
	void BVH::build( std::vector<TriangleMeshGeometry> * p_triangles )
	{
		std::cout << "Building BVH..." << std::endl;
		if ( p_triangles == nullptr || p_triangles->empty() )
		{
			throw std::logic_error( "BVH::build() error: no triangle provided" );
		}
		_triangles = p_triangles;

		Chrono chr;
		chr.start();

		_root = new BVHNode();
		_buildRec( _root, 0, _triangles->size(), 0 );

		chr.stop();

		std::cout << "[DONE]: " << chr.elapsedTime() << "s" << std::endl;
	}

	bool BVH::intersect( const Ray & p_ray, const float p_tMin, const float p_tMax, HitRecord & p_hitRecord ) const
	{
		p_hitRecord._distance = p_tMax;
		return _intersectRec( _root, p_ray, p_tMin, p_tMax, p_hitRecord );
	}

	bool BVH::intersectAny( const Ray & p_ray, const float p_tMin, const float p_tMax ) const
	{
		
		return _intersectAnyRec(_root, p_ray, p_tMin, p_tMax);
	}

	void BVH::_buildRec( BVHNode *			p_node,
						 const unsigned int p_firstTriangleId,
						 const unsigned int p_lastTriangleId,
						 const unsigned int p_depth )
	{
		p_node->_firstTriangleId = p_firstTriangleId;
		p_node->_lastTriangleId	 = p_lastTriangleId;

		for ( int i = p_firstTriangleId; i < p_lastTriangleId; i++ )
		{
			p_node->_aabb.extend( ( *_triangles )[ i ].getAABB() );
		}

		const unsigned int nbTriangle = p_lastTriangleId - p_firstTriangleId;
		if ( nbTriangle <= _maxTrianglesPerLeaf || p_depth >= _maxDepth ) return;

		float  bestCost	 = std::numeric_limits<float>::max();
		int	   bestAxis	 = -1;
		size_t bestSplit = p_firstTriangleId;

		// SAH
		for ( int axis = 0; axis < 3; axis++ )
		{
			std::sort( _triangles->begin() + p_firstTriangleId,
					   _triangles->begin() + p_lastTriangleId,
					   [ axis ]( const TriangleMeshGeometry & a, const TriangleMeshGeometry & b )
					   { return a.getAABB().centroid()[ axis ] < b.getAABB().centroid()[ axis ]; } );

			std::vector<AABB> leftAABBs( nbTriangle );
			std::vector<AABB> rightAABBs( nbTriangle );

			AABB leftBox, rightBox;
			for ( unsigned int i = 0; i < nbTriangle; i++ )
			{
				leftBox.extend( ( *_triangles )[ p_firstTriangleId + i ].getAABB() );
				leftAABBs[ i ] = leftBox;
			}
			for ( int i = nbTriangle - 1; i >= 0; i-- )
			{
				rightBox.extend( ( *_triangles )[ p_firstTriangleId + i ].getAABB() );
				rightAABBs[ i ] = rightBox;
			}

			for ( unsigned int i = 1; i < nbTriangle; i++ )
			{
				float costTemp = leftAABBs[ i - 1 ].area() * i + rightAABBs[ i ].area() * ( nbTriangle - i );
				if ( costTemp < bestCost )
				{
					bestCost  = costTemp;
					bestAxis  = axis;
					bestSplit = p_firstTriangleId + i;
				}
			}
		}

		if ( bestAxis == -1 || bestSplit == p_firstTriangleId || bestSplit == p_lastTriangleId ) return;

		std::sort( _triangles->begin() + p_firstTriangleId,
				   _triangles->begin() + p_lastTriangleId,
				   [ bestAxis ]( const TriangleMeshGeometry & a, const TriangleMeshGeometry & b )
				   { return a.getAABB().centroid()[ bestAxis ] < b.getAABB().centroid()[ bestAxis ]; } );

		p_node->_left = new BVHNode();
		_buildRec( p_node->_left, p_firstTriangleId, bestSplit, p_depth + 1 );

		p_node->_right = new BVHNode();
		_buildRec( p_node->_right, bestSplit, p_lastTriangleId, p_depth + 1 );
	}

	bool BVH::_intersectRec( const BVHNode * p_node,
							 const Ray &	 p_ray,
							 const float	 p_tMin,
							 const float	 p_tMax,
							 HitRecord &	 p_hitRecord ) const
	{
		if (p_node->_aabb.intersect(p_ray, p_tMin, p_tMax)) { 
			if (!p_node->isLeaf()) {
				
				bool b1 = _intersectRec( p_node->_left, p_ray, p_tMin, p_hitRecord._distance, p_hitRecord );
				bool b2 = _intersectRec( p_node->_right, p_ray, p_tMin, p_hitRecord._distance, p_hitRecord );
				
				return b1 || b2;
			}
			else { //si c'est une feuille on renvoie vrai
				Vec2f uvClosest;
				float tClosest = p_tMax;
				int	  indexClosest = -1; // index non initialisé
				for (int index = p_node->_firstTriangleId; index < p_node->_lastTriangleId; index++) {
					float t;
					Vec2f uv;
					if ( ( *_triangles )[ index ].intersect( p_ray, t, uv ) )
					{
						if ( t >= p_tMin && t < tClosest )
						{ 
							tClosest = t;
							indexClosest = index;
							uvClosest	 = uv;
						}
					}
				}
				if ( indexClosest == -1 ) { return false; }
				p_hitRecord._point	= p_ray.pointAtT( tClosest );
				p_hitRecord._normal = (*_triangles)[ indexClosest ].computeSmoothNormal( uvClosest );
				p_hitRecord.faceNormal( p_ray.getDirection() );
				p_hitRecord._distance = tClosest;
				p_hitRecord._object	  = (BaseObject*)(*_triangles)[indexClosest].getRefMesh();
				return true;
			}
		}
		return false;
	}

	bool BVH::_intersectAnyRec( const BVHNode * p_node,
								const Ray &		p_ray,
								const float		p_tMin,
								const float		p_tMax ) const
	{
		if( p_node->_aabb.intersect( p_ray, p_tMin, p_tMax ) )
		{
			if ( !p_node->isLeaf() )
			{
				bool b1 = _intersectAnyRec( p_node->_left, p_ray, p_tMin, p_tMax );
				bool b2 = _intersectAnyRec( p_node->_right, p_ray, p_tMin, p_tMax );

				return b1 || b2;
			}
			else
			{ // si c'est une feuille on renvoie vrai
				for ( int index = p_node->_firstTriangleId; index < p_node->_lastTriangleId; index++ )
				{
					float t;
					Vec2f uv;
					if ( ( *_triangles )[ index ].intersect( p_ray, t, uv ) )
					{
						if ( t >= p_tMin && t <= p_tMax )
						{ 
							return true;
						}
					}
				}
				return false;
			}
		}
		return false;
	}
} // namespace RT
