#ifndef __RT_ISICG_TRANSPARENT_MATERIAL__
#define __RT_ISICG_TRANSPARENT_MATERIAL__

#include "base_material.hpp"
#include "brdfs/lambert_brdf.hpp"

namespace RT
{
	class TransparentMaterial : public BaseMaterial
	{
	  public:
		TransparentMaterial( const std::string & p_name ) : BaseMaterial( p_name ), _brdf( BLACK ) {}
		TransparentMaterial( const std::string & p_name, const Vec3f & p_diffuse, const float & p_ior = 1.33f )
			: BaseMaterial( p_name ), _brdf( p_diffuse ), _ior( p_ior )
		{
		}

		virtual ~TransparentMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return BLACK;
		}

		inline const Vec3f & getFlatColor() const override { return BLACK; }

		const bool isTransparent() const override { return true; }

		const float getIOR() const override { return _ior; }

		MaterialType getType() const override { return MaterialType::TRANSPARENT;}

	  protected:
		LambertBRDF _brdf;
		float		_ior = 1.33f;
	};

} // namespace RT

#endif // __RT_ISICG_TRANSPARENT_MATERIAL__
