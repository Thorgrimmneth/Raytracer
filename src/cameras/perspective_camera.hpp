#ifndef __RT_ISICG_PERSPECTIVE_CAMERA__
#define __RT_ISICG_PERSPECTIVE_CAMERA__

#include "base_camera.hpp"

namespace RT
{
	class PerspectiveCamera : public BaseCamera
	{
	  public:
		PerspectiveCamera( const float p_aspectRatio );

		PerspectiveCamera( const Vec3f & p_position,
						   const Vec3f & p_lookAt,
						   const Vec3f & p_up,
						   const float	 p_fovy,
						   const float	 p_aspectRatio );

		~PerspectiveCamera() = default;

		Ray generateRay( const float p_sx, const float p_sy ) const override;

		inline Vec3f generateRandomPos() const;
		inline void	 setDefocusAngle( const float p_defocus_angle ) { _defocus_angle = p_defocus_angle; }
		inline void	 setFocalDistance( const float p_focalDistance ) { _focalDistance = p_focalDistance; }

	  private:
		void _updateViewport();

	  private:
		float _fovy			 = 60.f;
		float _focalDistance = 1.f;
		float _aspectRatio	 = 1.f;

		// Added for defocus blur
		float _defocus_angle = 0.f;

		// Local coordinates system
		Vec3f _u = Vec3f( 1.f, 0.f, 0.f );
		Vec3f _v = Vec3f( 0.f, 1.f, 0.f );
		Vec3f _w = Vec3f( 0.f, 0.f, -1.f );

		float viewportHeight = 0;
		float viewportWidth	 = 0;
		// Viewport data
		Vec3f _viewportTopLeftCorner = VEC3F_ZERO; // Top left corner position
		Vec3f _viewportU			 = VEC3F_ZERO; // Horizontal vector
		Vec3f _viewportV			 = VEC3F_ZERO; // Vertical vector
	};
} // namespace RT

#endif // __RT_ISICG_PERSPECTIVE_CAMERA__
