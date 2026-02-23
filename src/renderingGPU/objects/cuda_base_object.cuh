#include "../objectsUtils/cuda_aabb.cuh"

enum ObjectType{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT
};

struct BaseObject{
    float4 min;
    float4 max;
    ObjectType type;
};