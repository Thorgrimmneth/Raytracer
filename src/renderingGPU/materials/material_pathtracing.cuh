#include <curand_kernel.h>

enum MaterialPTType
{
    DIFFUSE,
    METAL,
    DIELECTRIC,
    EMISSIVE
};

struct SampleMatPT{
    float3 wi;
    float3 brdf;
    float pdf;
};

struct MaterialPT{
    MaterialPTType type;
    float3 albedo;
    float roughness;
    float ior;
    float3 emission;

    __device__
    SampleMatPT sample(float3 wo, float3 normal, curandState* states);
};