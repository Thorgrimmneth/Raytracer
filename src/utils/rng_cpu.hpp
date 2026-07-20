#pragma once

#include <random>

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