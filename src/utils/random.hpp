#ifndef __RT_ISICG_RANDOM__
#define __RT_ISICG_RANDOM__

#include <random>

namespace RT
{
	// Return a pseudo random float between 0 and 1
	static inline float randomFloat()
	{
		static std::mt19937							 gen;
		static std::uniform_real_distribution<float> dis( 0.f, 1.f );
		return dis( gen );
	}

	static inline double randomDouble()
	{
		static std::mt19937							  gen;
		static std::uniform_real_distribution<double> dis( 0.0, 1.0 );
		return dis( gen );
	}
} // namespace RT

#endif // __RT_ISICG_RANDOM__
