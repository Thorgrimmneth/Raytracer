#ifndef __RT_ISICG_QUAD_LIGHT__
#define __RT_ISICG_QUAD_LIGHT__

#include "base_light.hpp"

namespace RT
{
	class QuadLight : public BaseLight
	{
	  public:
		QuadLight( const std::string p_name,
				   const Vec3f &	 p_pos,
				   const Vec3f &	 p_u,
				   const Vec3f &	 p_v,
				   const Vec3f &	 p_color,
				   const float		 p_power = 1.f )
			: BaseLight( p_name, p_color, p_power ), _position( p_pos ), _u( p_u ), _v( p_v )
		{
			_normal = glm::cross( _u, _v );
			_area	= glm::length( _u ) * glm::length( _v )
					* glm::sin( glm::acos( glm::dot( _u, _v ) ) );
			_normal	   = glm::normalize( _normal );
			_isSurface = true;
		}
		virtual ~QuadLight() = default;

		LightSample sample( const Vec3f & p_point ) const override;

		Vec3f getPosition() { return _position; }
		Vec3f getU() { return _u; }
		Vec3f getV() { return _v; }
		Vec3f getNormal() {return _normal;}
		float getArea() {return _area;}
		LightType getType() const override { return LightType::QUAD;}
	  private:
		Vec3f _position; // bottom left corner
		Vec3f _u;
		Vec3f _v;
		Vec3f _normal;
		float _area;
	};
} // namespace RT
#endif // __RT_ISICG_QUAD_LIGHT__