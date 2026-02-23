#include "../objectsUtils/cuda_aabb.cuh"

enum ObjectType{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT
};

struct BaseObject{
    AABB bbox;
    ObjectType type;
};