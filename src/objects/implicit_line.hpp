#ifndef __RT_ISICG_IMPLICIT_LINE__
#define __RT_ISICG_IMPLICIT_LINE__

#include "implicit_surface.hpp"

namespace RT
{
	class ImplicitLine : public ImplicitSurface
	{
	  public:
		ImplicitLine( const std::string & p_name, const Vec3f & p_a, const Vec3f & p_b, const float & p_radius )
			: ImplicitSurface( p_name ), _a( p_a ), _b( p_b ), _radius( p_radius )
		{
		}

		~ImplicitLine() = default;
		ObjectType getType() const override { return ObjectType::Implicit;}
	  private:
		virtual float _sdf( const Vec3f & p_point ) const override
		{
			Vec3f pa = p_point - _a;
			Vec3f ba = _b - _a;
			float h	 = glm::clamp( glm::dot( pa, ba ) / glm::dot( ba, ba ), 0.f, 1.f );
			return glm::length( pa - h * ba ) - _radius;
		}
		Vec3f _a;
		Vec3f _b;
		float _radius;
	};

} // namespace RT

#endif // __RT_ISICG_IMPLICIT_LINE__
