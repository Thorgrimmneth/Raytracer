#ifndef __RT_ISICG_DEFINES__
#define __RT_ISICG_DEFINES__

#include <iostream>
#include <limits>
#include <random>
#include <string>

// Scalars.
const float PIf = 3.14159265358979323846f;

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

#endif // __RT_ISICG_DEFINES__
