#include "sphere_geometry.hpp"

namespace RT
{
	bool SphereGeometry::intersect( const Ray & p_ray, float & p_t1, float & p_t2 ) const
	{
		
		Vec3f		current_center = getCenter( p_ray.getTime() );
		const Vec3f oc			   = p_ray.getOrigin() - current_center;
		float		a  = glm::dot( p_ray.getDirection(), p_ray.getDirection() );
		float		b  = 2 * glm::dot( p_ray.getDirection(), oc );
		float		c  = glm::dot( oc, oc ) - _radius * _radius;

		float delta = b * b - 4 * a * c;
		if ( delta < 0 ) { return false; }

		float intersection1 = ( -b - glm::sqrt( delta ) ) / ( 2 * a );
		float intersection2 = ( -b + glm::sqrt( delta ) ) / ( 2 * a );

		p_t1 = intersection1;
		p_t2 = intersection2;

		return true;
	}

} // namespace RT
