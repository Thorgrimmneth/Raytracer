#include "cylinder_light.hpp"
#include "utils/random.hpp"

namespace RT
{
	LightSample CylinderLight::sample( const Vec3f & p_point ) const
	{
		float u = randomFloat();
		float v = randomFloat();

		Vec3f uVec = glm::normalize( glm::cross( _direction, Vec3f( 1, 0, 0 ) ) );
		if ( glm::length( uVec ) < 1e-3f ) uVec = glm::normalize( glm::cross( _direction, Vec3f( 0, 1, 0 ) ) );

		Vec3f vVec = glm::normalize( glm::cross( _direction, uVec ) );

		float z	  = u * _height;
		float phi = v * 2.f * PIf;

		Vec3f pointOnCircle = _radius * ( std::cos( phi ) * uVec + std::sin( phi ) * vVec );
		Vec3f randomPos		= _position + z * _direction + pointOnCircle;

		Vec3f axisPoint = _position + z * _direction;
		Vec3f normal	= normalize( randomPos - axisPoint );
		if ( _isReversed ) normal = -normal; // we want the normal to point inside the cylinder if the light is reversed (used to light up the laser beam)

		Vec3f direction = normalize( randomPos - p_point );
		float distance	= glm::distance( p_point, randomPos );

		float cosTheta = glm::dot( normal, -direction );
		if ( cosTheta <= 0.f ) return LightSample();

		float pdf = ( distance * distance ) / ( _area * cosTheta );

		Vec3f		radiance = ( _color * _power ) / pdf;
		LightSample sample;
		sample._radiance  = radiance;
		sample._pdf		  = pdf;
		sample._power	  = _power;
		sample._distance  = distance;
		sample._direction = direction;

		return sample;
	}

} // namespace RT