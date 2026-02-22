#include "cuda_hitrecord.cuh"

__device__ 
void HitRecord::faceNormal(const float3 &direction)
{
    normal = dot(direction, normal) < 0.f ? normal : -normal; 
}
