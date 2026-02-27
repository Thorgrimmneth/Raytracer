#ifndef __RT_ISICG_METAL_MATERIAL__
#define __RT_ISICG_METAL_MATERIAL__

#include "base_material.hpp"
#include "brdfs/cook_torrance_brdf.hpp"
#include "brdfs/lambert_brdf.hpp"

namespace RT
{
	class MetalMaterial : public BaseMaterial
	{
	  public:
		MetalMaterial() : BaseMaterial( "default" ), _brdf( WHITE ), _cookTorranceBrdf( WHITE, 0.5f ) {}
		MetalMaterial( const std::string & p_name,
					   const Vec3f &	   p_diffuse,
					   const float &	   p_ruggedness,
					   const float &	   p_metalness )
			: BaseMaterial( p_name ), _brdf( p_diffuse ), _cookTorranceBrdf( p_diffuse, p_ruggedness ),
			  _metalness( p_metalness )
		{
		}

		MetalMaterial( const std::string & p_name)
			: BaseMaterial( p_name )
		{
			_brdf = LambertBRDF( Vec3f( randomFloat(), randomFloat(), randomFloat() ) );
			_cookTorranceBrdf = CookTorranceBRDF( _brdf.getKd(), randomFloat() );
			_metalness		  = randomFloat();
		}


		virtual ~MetalMaterial() = default;

		Vec3f shade( const Ray &		 p_ray,
					 const HitRecord &	 p_hitRecord,
					 const LightSample & p_lightSample ) const override
		{
			return _brdf.evaluate() * ( 1.f - _metalness )
				   + _cookTorranceBrdf.evaluate( p_ray.getDirection(), p_hitRecord._normal, p_lightSample._direction )
						 * _metalness;
		}

		inline const Vec3f & getFlatColor() const override { return _brdf.getKd(); }
		inline float getMetalness() const { return _metalness;}
		inline float getRuggedness() const {return _cookTorranceBrdf.getR();}
		MaterialType getType() const override { return MaterialType::METAL;}

	  protected:
		LambertBRDF		 _brdf;
		CookTorranceBRDF _cookTorranceBrdf;
		float			 _metalness = 0.5f;
	};

} // namespace RT

#endif // __RT_ISICG_METAL_MATERIAL__
