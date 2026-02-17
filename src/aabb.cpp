#include "aabb.hpp"

namespace RT
{
	bool AABB::intersect( const Ray & p_ray, const float p_tMin, const float p_tMax ) const
	{
		const Vec3f origin	  = p_ray.getOrigin();
		const Vec3f direction = p_ray.getDirection();
		const Vec3f invdir	  = p_ray.getInvDirection();
		float		tmin, tmax, tymin, tymax, tzmin, tzmax;

		if ( invdir.x >= 0 )
		{
			tmin = ( _min.x - origin.x ) / direction.x; // calcul pour x
			tmax = ( _max.x - origin.x ) / direction.x;
		}
		else
		{
			tmin = ( _max.x - origin.x ) / direction.x;
			tmax = ( _min.x - origin.x ) / direction.x;
		}

		if ( invdir.y >= 0 )
		{
			tymin = ( _min.y - origin.y ) / direction.y; // calcul pour y
			tymax = ( _max.y - origin.y ) / direction.y;
		}
		else
		{
			tymin = ( _max.y - origin.y ) / direction.y;
			tymax = ( _min.y - origin.y ) / direction.y;
		}

		if ( ( tmin > tymax ) || ( tymin > tmax ) ) return false;

		if ( tymin > tmin ) { tmin = tymin; }
		if ( tymax < tmax ) { tmax = tymax; }

		if ( invdir.z >= 0 )
		{
			tzmin = ( _min.z - origin.z ) / direction.z; // calcul pour z
			tzmax = ( _max.z - origin.z ) / direction.z;
		}
		else
		{
			tzmin = ( _max.z - origin.z ) / direction.z;
			tzmax = ( _min.z - origin.z ) / direction.z;
		}

		if ( ( tmin > tzmax ) || ( tzmin > tmax ) ) return false;

		if ( tzmin > tmin ) { tmin = tzmin; }
		if ( tzmax < tmax ) { tmax = tzmax; }

		if ( ( tmin < p_tMin ) && ( tmax > p_tMax ) ) return false;
		return true;
	}
} // namespace RT
