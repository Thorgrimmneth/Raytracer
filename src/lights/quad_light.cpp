#include "quad_light.hpp"

namespace RT
{
	LightSample QuadLight::sample( const Vec3f & p_point ) const
	{
		Vec3f randomPos		  = _position + randomFloat() * _u + randomFloat() * _v;
		Vec3f direction		  = glm::normalize( randomPos - p_point );
		float distance		  = glm::distance( p_point, randomPos );
		float angle			  = glm::dot( _normal, direction );
		float geometricFactor = ( distance * distance ) / angle;
		float PDF			  = geometricFactor / _area;

		Vec3f radiance = ( _color * _power ) / PDF;

		LightSample rep;
		rep._radiance  = radiance;
		rep._pdf	   = PDF;
		rep._power	   = _power;
		rep._distance  = distance;
		rep._direction = direction;

		return rep;
	}
} // namespace RT