#ifndef __RT_ISICG_CYLINDER_LIGHT__
#define __RT_ISICG_CYLINDER_LIGHT__

#include "base_light.hpp"

namespace RT
{
	class CylinderLight : public BaseLight
	{
	  public:
		CylinderLight( const std::string p_name,
					   const Vec3f &	 p_pos,
					   const float &	 p_height,
					   const float &	 p_radius,
					   const Vec3f &	 p_direction,
					   const Vec3f &	 p_color,
					   const float &	 p_power	  = 1.f,
					   const bool &		 p_isReversed = false )
			: BaseLight( p_name, p_color, p_power ), _position( p_pos ), _height( p_height ), _radius( p_radius ),
			  _direction( normalize( p_direction ) ), _isReversed( p_isReversed )
		{
			_area	   = 2.f * PIf * p_radius * p_height;
			_isSurface = true;
		}
		virtual ~CylinderLight() = default;

		LightSample sample( const Vec3f & p_point ) const override;

		Vec3f getPosition() { return _position; }
		float getHeight() { return _height; }
		float getRayon() { return _radius; }
		Vec3f getDirection() { return _direction;}
		float getArea(){return _area;}
		LightType getType() const override { return LightType::CYLINDER;}
	  private:
		Vec3f _position;
		float _height;
		float _radius;
		Vec3f _direction;
		float _area;
		bool  _isReversed; // if true, the light is emitted inside the cylinder
	};
} // namespace RT
#endif // __RT_ISICG_CYLINDER_LIGHT__