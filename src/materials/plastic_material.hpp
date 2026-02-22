#ifndef __RT_ISICG_PLASTIC_MATERIAL__
#define __RT_ISICG_PLASTIC_MATERIAL__

#include "base_material.hpp"
#include "brdfs/blinn_phong_brdf.hpp"
#include "brdfs/lambert_brdf.hpp"

namespace RT
{
	class PlasticMaterial : public BaseMaterial
	{
	  public:
		PlasticMaterial( const std::string & p_name,
						 const Vec3f &		 p_diffuse,
						 const float &		 p_diffuse_value,
						 const float &		 p_s )
			: BaseMaterial( p_name ), _brdf( p_diffuse * p_diffuse_value ),
			  _blinnPhongBrdf( p_diffuse * ( 1 - p_diffuse_value ), p_s )
		{
		}

		virtual ~PlasticMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return _brdf.evaluate()
				   + _blinnPhongBrdf.evaluate( p_ray.getDirection(), p_hitRecord._normal, p_lightSample._direction );
		}

		inline const Vec3f & getFlatColor() const override { return _brdf.getKd(); }
		inline float getShininess() const { return _s;}

		MaterialType getType() const override { return MaterialType::PLASTIC;}

	  protected:
		LambertBRDF	   _brdf;
		BlinnPhongBRDF _blinnPhongBrdf;
		float		   _s = 8.f;
	};

} // namespace RT

#endif // __RT_ISICG_PLASTIC_MATERIAL__
