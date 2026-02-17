#ifndef __RT_ISICG_BRDF_LAMBERT__
#define __RT_ISICG_BRDF_LAMBERT__

#include "defines.hpp"

namespace RT
{
	class LambertBRDF
	{
	  public:
		LambertBRDF() : _kd( WHITE ) {};
		LambertBRDF( const Vec3f & p_kd ) : _kd( p_kd ) {};

		// formula : _kd / PI
		inline Vec3f evaluate() const { return _kd * INV_PIf; }

		inline const Vec3f & getKd() const { return _kd; }

	  private:
		Vec3f _kd = WHITE;
	};
} // namespace RT

#endif // __RT_ISICG_BRDF_LAMBERT__
