#ifndef __RT_ISICG_DEFINES__
#define __RT_ISICG_DEFINES__

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <string>

// Scalars.
const float PIf = 3.14159265358979323846f;

// Vectors.
using Vec3f = glm::vec3;
using Vec4f = glm::vec4;

const Vec3f VEC3F_ZERO = Vec3f(0.f);
const Vec4f VEC4F_ZERO = Vec4f(0.f);

// Paths.
const std::string RESULTS_PATH = "results/images/";

// Utils.
static inline float intAsFloat(const int p_i)
{
    union {
        int a;
        float b;
    } u;
    u.a = p_i;
    return u.b;
}
static inline int floatAsInt(const float p_f)
{
    union {
        float a;
        int b;
    } u;
    u.a = p_f;
    return u.b;
}

inline float degToRad(float deg) { return deg * PIf / 180.f; }
inline float radToDeg(float rad) { return rad * 180.f / PIf; }
inline float length(Vec3f a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

inline std::mt19937 &globalRNG()
{
    static std::mt19937 gen(42);
    return gen;
}

inline void setSeed(uint32_t seed) { globalRNG().seed(seed); }

inline double randomDouble()
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(globalRNG());
}

inline double randomDouble(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(globalRNG());
}

inline float randomFloat() { return static_cast<float>(randomDouble()); }

inline float randomFloat(float min, float max) { return static_cast<float>(randomDouble(min, max)); }

#endif // __RT_ISICG_DEFINES__
