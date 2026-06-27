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

#endif // __RT_ISICG_DEFINES__
