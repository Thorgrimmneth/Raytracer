#include "implicitSphere.cuh"



    /*
    __device__
    float3 ImplicitSphere::computeNormal(const float3 &point, const double time) const
    {
        float e = 1.e-4f;
        return normalize(make_float3(1.f, -1.f, -1.f) * sdf(point + make_float3(e, -e, -e))
                       + make_float3(-1.f, -1.f, 1.f) * sdf(point + make_float3(-e, -e, e))
                       + make_float3(-1.f, 1.f, -1.f) * sdf(point + make_float3(-e, e, -e))
                       + make_float3(1.f ,1.f, 1.f) * sdf(point + make_float3(e, e, e)));
    }*/

    

    