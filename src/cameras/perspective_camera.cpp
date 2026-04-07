#include "perspective_camera.hpp"
#include <glm/gtx/string_cast.hpp>

namespace RT
{

	PerspectiveCamera::PerspectiveCamera( const float p_aspectRatio ) : _aspectRatio( p_aspectRatio )
	{
		_updateViewport();
	}

	PerspectiveCamera::PerspectiveCamera( const Vec3f & p_position,
										  const Vec3f & p_lookAt,
										  const Vec3f & p_up,
										  const float	p_fovy,
										  const float	p_aspectRatio )
		: BaseCamera( p_position ), _fovy( p_fovy ), _aspectRatio( p_aspectRatio )
	{
		_w = glm::normalize( p_position - p_lookAt );
		_u = glm::normalize( glm::cross( p_up, _w ) );
		_v = glm::normalize( glm::cross( _w, _u ) );
		_updateViewport();
	}

	void PerspectiveCamera::_updateViewport()
	{
		viewportHeight		   = 2.f * glm::tan( glm::radians( _fovy / 2.f ) ) * _focalDistance;
		viewportWidth		   = viewportHeight * _aspectRatio;
		_viewportV			   = _v * viewportHeight;
		_viewportU			   = _u * viewportWidth;
		_viewportTopLeftCorner = _position - _w * _focalDistance + _viewportV / 2.f - _viewportU / 2.f;
	}

	// Generates a random point in a disk (z = 0 because the lens is in 2D)
	Vec3f PerspectiveCamera::generateRandomPos() const
	{
		while ( true )
		{
			Vec3f p = Vec3f( randomFloat( -1.f, 1.f ), randomFloat( -1.f, 1.f ), 0.f );
			if ( glm::length( p ) <= 1.f ) { return p; }
		}
	}

	Ray PerspectiveCamera::generateRay( const float p_sx, const float p_sy ) const
	{
		Vec3f rayTarget = _viewportTopLeftCorner + p_sx * _viewportU - p_sy * _viewportV;

		// If the defocus angle is 0, we can use the ray directly
		if ( abs( _defocus_angle ) <= 0.e-4f )
		{
			Vec3f rayDirection = normalize( rayTarget - _position );
			return Ray( _position, rayDirection );
		}

		Vec3f direction = normalize( rayTarget - _position );

		Vec3f focusPoint = _position + direction * _focalDistance;

		float lensRadius = _focalDistance * glm::tan( glm::radians( _defocus_angle / 2.f ) );

		// Generate a random point in the lens disk
		Vec3f randDisk	 = generateRandomPos() * lensRadius;

		// Generate a random point on the lens (world space)
		Vec3f lensOrigin = _position + randDisk.x * _u - randDisk.y * _v;

		double ray_time = randomDouble();
		return Ray( lensOrigin, normalize( focusPoint - lensOrigin ), ray_time);
	}

} // namespace RT
