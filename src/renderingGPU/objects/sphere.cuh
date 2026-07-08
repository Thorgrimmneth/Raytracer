#pragma once

struct Sphere
{
    float3 center1;
    float radius;
    int materialIndex;

    Sphere() : center1(make_float3(0.f)), radius(1.f), materialIndex(0) {}
    Sphere(const float3 &c, float r, int m = 0) : center1(c), radius(r), materialIndex(m) {}
};